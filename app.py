#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageLab —— 本地生图工作台（当前引擎：Qwen-Image 2.1）

功能：
  · 文生图 / 图像编辑（参考图可来自本机文件或刚生成的图）
  · 文本编码器热切换（W4A8 / INT8）
  · 批量出图（同参不同种子）
  · 提示词历史（服务端 history.json）

用法：双击 启动_ImageLab.bat
调试：python app.py --no-window [--comfy-home <ComfyUI 目录>]
"""
from __future__ import annotations

import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_comfy_home() -> str:
    """定位 ComfyUI 安装目录。优先级：
    --comfy-home <path> 参数 > IMAGELAB_COMFY_HOME 环境变量 > ImageLab 同级的常见目录。
    不硬编码路径：别人拿到这份代码，把 ImageLab 放到自己的 ComfyUI 旁边就能跑。"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--comfy-home":
            if i + 1 >= len(argv):
                raise SystemExit("--comfy-home 后面要跟 ComfyUI 目录")
            return os.path.abspath(argv[i + 1])
        if a.startswith("--comfy-home="):
            return os.path.abspath(a.split("=", 1)[1])
    env = os.environ.get("IMAGELAB_COMFY_HOME")
    if env:
        return os.path.abspath(env)
    parent = os.path.dirname(APP_DIR)
    for cand in ("ComfyUI_windows_portable", "ComfyUI"):
        p = os.path.join(parent, cand)
        if os.path.isfile(os.path.join(p, "ComfyUI", "main.py")) \
                or os.path.isfile(os.path.join(p, "main.py")):
            return os.path.abspath(p)
    # 找不到也返回便携版风格的路径，让 boot_comfy 报出明确错误而不是静默失败
    return os.path.abspath(os.path.join(parent, "ComfyUI_windows_portable"))


COMFY_HOME = _resolve_comfy_home()


def _comfy_root() -> str:
    """真正装着 main.py / models / input / output 的目录：
    便携版是 home 下面那层 ComfyUI 子目录，源码版 ComfyUI 就是 home 本身。"""
    nested = os.path.join(COMFY_HOME, "ComfyUI")
    return nested if os.path.isfile(os.path.join(nested, "main.py")) else COMFY_HOME


COMFY_ROOT = _comfy_root()
COMFY_MAIN = os.path.join(COMFY_ROOT, "main.py")
# 起 ComfyUI 用哪份 python：便携版用自带 python_embeded，源码版用当前解释器
COMFY_PY = next((p for p in (os.path.join(COMFY_HOME, "python_embeded", "python.exe"),
                             sys.executable) if os.path.isfile(p)), sys.executable)
MODELS_DIR = os.path.join(COMFY_ROOT, "models")
COMFY_IN = os.path.join(COMFY_ROOT, "input")
COMFY_OUT = os.path.join(COMFY_ROOT, "output")
COMFY_LOG = os.path.join(APP_DIR, "comfyui.log")

COMFY_PORT = 8188
COMFY = "http://127.0.0.1:%d" % COMFY_PORT
UI_HOST, UI_PORT = "127.0.0.1", 8765

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

APP_NAME = "ImageLab"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 引擎注册表
#
# 接新模型 = 在这里加一项。字段说明：
#   builder : 用哪个构图函数。同架构换权重就复用现有的；换架构（如 Flux）新增一个。
#   unet    : 候选权重文件名，按顺序取第一个存在的（找不到即视为「未安装」）
#   defaults: 该引擎的默认尺寸/步数，前端切引擎时用
#   caps    : 能力位，前端据此显示/隐藏控件
#
# 所有引擎共用同一套文本编码器 / VAE（见 scan_models），换架构时再扩展。
ENGINE_DEFS = {
    "qwen-image-2.1": {
        "name": "Qwen-Image 2.1",
        "vendor": "Alibaba Qwen",
        "builder": "qwen_image_21",
        "unet": ["qwen_image_2.1_int8_convrot.safetensors"],
        "min_bytes": 6_500_000_000,
        "defaults": {"width": 1024, "height": 1024, "steps": 20},
        "caps": {"refs": True, "encoder": True, "pe": True},
        "note": "标准版，20–25 步，质量优先",
    },
    "qwen-image-2.1-turbo": {
        "name": "Qwen-Image 2.1 Turbo",
        "vendor": "Alibaba Qwen",
        "builder": "qwen_image_21",
        "unet": ["qwen_image_2.1_turbo_int8_convrot.safetensors"],
        "min_bytes": 6_500_000_000,
        "defaults": {"width": 1024, "height": 1024, "steps": 8},
        "caps": {"refs": True, "encoder": True, "pe": True},
        "note": "蒸馏少步版，4–8 步出图，速度快",
    },
}


def _precision(name: str) -> str:
    low = (name or "").lower()
    for tag, label in (("int8", "INT8"), ("fp8", "FP8"), ("bf16", "BF16"),
                       ("fp16", "FP16"), ("fp32", "FP32")):
        if tag in low:
            return label
    return ""


def resolve_engine(engine_id: str | None = None) -> dict | None:
    """把引擎 id 解析成完整描述，包含权重是否真的存在。"""
    eid = engine_id or STATE.get("engine") or next(iter(ENGINE_DEFS))
    spec = ENGINE_DEFS.get(eid)
    if not spec:
        return None
    folder = os.path.join(MODELS_DIR, "diffusion_models")
    unet = ""
    for name in spec["unet"]:
        path = os.path.join(folder, name)
        # 只判断存在不够：下载中的半截文件也会被当成已安装，加载时才报错
        try:
            if os.path.getsize(path) >= spec.get("min_bytes", 500_000_000):
                unet = name
                break
        except OSError:
            continue
    return {
        "id": eid,
        "name": spec["name"],
        "vendor": spec["vendor"],
        "note": spec.get("note", ""),
        "installed": bool(unet),
        "precision": _precision(unet),
        "weights": unet,
        "clip": STATE["models"].get("clip") or "",
        "vae": STATE["models"].get("vae") or "",
        "defaults": spec.get("defaults", {}),
        "caps": spec.get("caps", {}),
    }


def list_engines() -> list:
    return [resolve_engine(eid) for eid in ENGINE_DEFS]


def engine_info() -> dict:
    return resolve_engine() or {}

# 历史记录存在服务端文件里，不再用 localStorage。
# 原因：localStorage 按 origin（含端口）隔离，端口一变历史就"消失"。
HISTORY_FILE = os.path.join(APP_DIR, "history.json")
HISTORY_MAX = 100
OUTPUT_PREFIX = "imagelab"

# PE 增强的中文系统提示词。
# 不指定时模型默认输出英文，且会把「」里的汉字翻译掉——实测会毁掉中文文字渲染。
PE_SYSTEM_PROMPT = (
    "你是文生图提示词专家。把用户的简短描述扩写成一段具体、可拍摄的中文画面描述，"
    "包含主体、动作、环境、光线、构图、镜头与风格。"
    "用「」括起的文字必须原样保留，不要翻译、不要改写。只输出描述本身。"
)
UI_HTML = os.path.join(APP_DIR, "ui.html")

STATE = {"owned": False, "proc": None, "ready": False, "error": None,
         "models": {}, "encoders": [], "engine": None}
STATE["engine"] = next(iter(ENGINE_DEFS))
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
LAST_BEAT = time.time()


# ------------------------------------------------------------------ 基础

def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + "  " + msg
    print(line, flush=True)
    try:
        with open(os.path.join(APP_DIR, "imagelab.log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d ") + line + "\n")
    except Exception:
        pass


def http_json(url: str, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body.decode("utf-8")) if body else {}


def comfy_stats(timeout=2.0):
    try:
        return http_json(COMFY + "/system_stats", timeout=timeout)
    except Exception:
        return None


def vram_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=CREATE_NO_WINDOW).stdout.strip()
        used, total = [int(x) for x in out.split(",")[:2]]
        return used, total
    except Exception:
        return None


def safe_name(name: str) -> str:
    """只保留文件名部分，去掉路径穿越和奇怪字符。"""
    base = os.path.basename(name or "")
    base = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fa5]", "_", base)
    return base[:120] or "ref.png"


def scan_models() -> None:
    def pick(subdir, preferred):
        folder = os.path.join(MODELS_DIR, subdir)
        try:
            names = [f for f in os.listdir(folder) if f.endswith(".safetensors")]
        except OSError:
            return None, []
        chosen = next((w for w in preferred if w in names), names[0] if names else None)
        return chosen, names

    unet, _ = pick("diffusion_models", ["qwen_image_2.1_int8_convrot.safetensors"])
    vae, _ = pick("vae", ["qwen_image_2.1_vae_bf16.safetensors"])
    clip, clips = pick("text_encoders",
                       ["qwen3vl_8b_w4a8.safetensors", "qwen3vl_8b_int8_convrot.safetensors"])

    # 只暴露 2.1 用的文本编码器，PE 增强模型排除
    encs = [c for c in sorted(clips) if c.startswith("qwen3vl_8b_")]
    order = {"qwen3vl_8b_w4a8.safetensors": 0, "qwen3vl_8b_int8_convrot.safetensors": 1}
    encs.sort(key=lambda c: (order.get(c, 9), c))

    def find_pe(key):
        cands = [n for n in clips if key in n]
        if not cands:
            return None
        cands.sort(key=lambda n: 0 if "w4a8" in n else 1)
        return cands[0]

    with LOCK:
        STATE["models"] = {"unet": unet, "clip": clip, "vae": vae,
                           "pe_t2i": find_pe("pe_t2i"), "pe_i2i": find_pe("pe_i2i")}
        STATE["encoders"] = encs


# ------------------------------------------------------------------ 启动 ComfyUI

def boot_comfy() -> None:
    try:
        if comfy_stats():
            log("ComfyUI 已在运行，复用")
            STATE["ready"] = True
            scan_models()
            return
        if not os.path.isfile(COMFY_MAIN):
            raise RuntimeError(
                "找不到 ComfyUI（预期在 %s）。用 --comfy-home 参数或 IMAGELAB_COMFY_HOME 环境变量指定它的安装目录。"
                % COMFY_MAIN)
        log("启动 ComfyUI 后端 … (%s)" % COMFY_HOME)
        logf = open(COMFY_LOG, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            [COMFY_PY, "-s", COMFY_MAIN, "--port", str(COMFY_PORT),
             "--listen", "127.0.0.1", "--disable-auto-launch", "--reserve-vram", "0.7"],
            cwd=COMFY_HOME, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW)
        STATE["proc"] = proc
        STATE["owned"] = True
        for _ in range(240):
            if proc.poll() is not None:
                raise RuntimeError("ComfyUI 进程已退出，详见 comfyui.log")
            if comfy_stats(timeout=1.0):
                STATE["ready"] = True
                log("ComfyUI 就绪")
                scan_models()
                return
            time.sleep(1)
        raise RuntimeError("ComfyUI 启动超时（240 秒）")
    except Exception as exc:                      # noqa: BLE001
        STATE["error"] = str(exc)
        log("后端启动失败: %s" % exc)


def shutdown_comfy() -> None:
    proc = STATE.get("proc")
    if STATE.get("owned") and proc and proc.poll() is None:
        log("关闭 ComfyUI")
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# ------------------------------------------------------------------ 构图

def _loader_nodes(clip: str, refs: list[str], unet: str = "") -> tuple[dict, dict]:
    """返回 (公共 loader 节点, TextEncode 的 images 输入)。

    注意：Autogrow 类型在 API 格式里必须写成扁平的 "images.image_1"，
    写成嵌套 dict（{"images": {"image_1": ...}}）会被静默忽略——参考图就白传了。
    """
    nodes = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet or STATE["models"]["unet"],
                         "weight_dtype": "default"}},
        "2": {"class_type": "QwenImage21Cache",
              "inputs": {"model": ["1", 0], "device": "auto", "dtype": "int8"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": clip, "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": STATE["models"]["vae"]}},
    }
    images = {}
    for i, name in enumerate(refs[:10], start=1):
        nid = str(20 + i)
        nodes[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        images["images.image_%d" % i] = [nid, 0]
    return nodes, images


def build_qwen_image_21(prompt: str, width: int, height: int, steps: int, seed: int,
                        clip: str, refs: list[str], prefix: str,
                        resolution: int = 1024, unet: str = "") -> dict:
    """Qwen-Image 2.1 架构的构图（标准版和 turbo 共用，只换权重）。"""
    nodes, images = _loader_nodes(clip, refs, unet)
    enc = {"clip": ["3", 0], "prompt": prompt, "negative_prompt": "",
           "vae": ["4", 0], "resolution": max(256, min(2048, int(resolution or 1024)))}
    if images:
        enc.update(images)
    nodes["5"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}

    if images:
        # 编辑：latent 由参考图尺寸决定
        latent = ["5", 2]
    else:
        nodes["6"] = {"class_type": "EmptyLatentImage",
                      "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["6", 0]

    nodes["7"] = {"class_type": "KSampler",
                  "inputs": {"model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1],
                             "latent_image": latent, "seed": seed, "steps": steps, "cfg": 1,
                             "sampler_name": "euler", "scheduler": "simple", "denoise": 1}}
    nodes["8"] = {"class_type": "VAEDecode",
                  "inputs": {"samples": ["7", 0], "vae": ["4", 0]}}
    nodes["9"] = {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return nodes


BUILDERS = {"qwen_image_21": build_qwen_image_21}


def build_graph(engine_id: str, prompt: str, width: int, height: int, steps: int, seed: int,
                clip: str, refs: list[str], prefix: str, resolution: int = 1024) -> dict:
    """按引擎分发构图。"""
    info = resolve_engine(engine_id)
    if not info:
        raise RuntimeError("未知引擎：%s" % engine_id)
    if not info["installed"]:
        raise RuntimeError("引擎「%s」未安装，缺少权重：%s"
                           % (info["name"], " 或 ".join(ENGINE_DEFS[engine_id]["unet"])))
    builder = BUILDERS.get(ENGINE_DEFS[engine_id].get("builder", ""))
    if not builder:
        raise RuntimeError("引擎「%s」缺少构图函数" % info["name"])
    return builder(prompt, width, height, steps, seed, clip, refs, prefix,
                   resolution=resolution, unet=info["weights"])

def watch_vram(job: dict) -> None:
    while not job.get("done"):
        v = vram_mib()
        if v:
            job["peak"] = max(job.get("peak") or 0, v[0])
        time.sleep(1.0)


class Cancelled(Exception):
    """用户点了停止。不当作错误处理，图能留多少留多少。"""


def submit_one(graph: dict) -> str:
    res = http_json(COMFY + "/prompt", {"prompt": graph, "client_id": "imagelab"}, timeout=60)
    pid = res.get("prompt_id")
    if not pid:
        raise RuntimeError("提交失败: %s" % json.dumps(res, ensure_ascii=False)[:400])
    return pid


def comfy_interrupt() -> bool:
    """停掉 ComfyUI 当前正在跑的任务。

    只用 /interrupt 不够：它只打断已在执行的节点，而「提交 → 开始执行」之间
    有几百毫秒窗口，落在窗口里的任务会照常跑完。所以再清一次排队队列。
    这个 ComfyUI 实例归 ImageLab 独用，清空队列不会误伤别人。
    """
    ok = False
    for path, payload in (("/interrupt", {}), ("/queue", {"clear": True})):
        try:
            http_json(COMFY + path, payload, timeout=10)
            ok = True
        except Exception as exc:                  # noqa: BLE001
            log("停止请求 %s 失败：%s" % (path, exc))
    return ok


def peek_output(pid: str):
    """中断和出图撞车时，把已经落盘的那张捞回来，别白算。"""
    try:
        entry = http_json(COMFY + "/history/%s" % pid, timeout=5).get(pid) or {}
        for node in (entry.get("outputs") or {}).values():
            for img in node.get("images", []):
                return {"filename": img.get("filename"),
                        "subfolder": img.get("subfolder", "")}
    except Exception:                             # noqa: BLE001
        pass
    return None


def wait_one(pid: str, timeout=900, cancel=None) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.4)
        if cancel is not None and cancel.is_set():
            # 停止信号可能刚好落在最后一帧之后：先看看图有没有已经存下来
            done = peek_output(pid)
            if done:
                return done
            raise Cancelled()
        entry = http_json(COMFY + "/history/%s" % pid, timeout=10).get(pid)
        if not entry:
            continue
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            # 中断也是以 error 记进 history 的，别把它报成"执行出错"
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            raise RuntimeError(_error_text(status.get("messages") or []))
        for node in (entry.get("outputs") or {}).values():
            for img in node.get("images", []):
                return {"filename": img.get("filename"),
                        "subfolder": img.get("subfolder", "")}
        if status.get("completed"):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            raise RuntimeError("任务结束但没有输出图片")
    raise RuntimeError("等待超时")


def build_pe_graph(prompt: str, kind: str, refs: list[str], thinking: bool = False) -> dict:
    """PE 提示词增强图：CLIPLoader(PE) → TextGenerate → PreviewAny。

    PreviewAny 会把文本放进 history 的 outputs.text，用它取回增强结果。
    """
    pe = STATE["models"].get("pe_" + kind) or STATE["models"].get("pe_t2i")
    nodes = {
        "1": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": pe, "type": "stable_diffusion", "device": "default"}},
        "2": {"class_type": "TextGenerate", "inputs": {
            "clip": ["1", 0], "prompt": prompt, "max_length": 1024,
            "sampling_mode": "on",
            "sampling_mode.temperature": 0.3 if kind == "i2i" else 0.7,
            "sampling_mode.top_k": 20,
            "sampling_mode.top_p": 0.95,
            "sampling_mode.min_p": 0.0,
            "sampling_mode.repetition_penalty": 1.05,
            "sampling_mode.seed": 42,
            "sampling_mode.presence_penalty": 1.5,
            "thinking": thinking,
            "use_default_template": True, "mtp": "auto",
            "system_prompt": PE_SYSTEM_PROMPT}},
        "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
    }
    if kind == "i2i" and refs:
        nodes["4"] = {"class_type": "LoadImage", "inputs": {"image": refs[0]}}
        nodes["2"]["inputs"]["image"] = ["4", 0]
    return nodes


def looks_broken(text: str) -> bool:
    """PE 偶发退化（重复词、过短）。检测到就丢弃，不覆盖用户原文。"""
    if len(text) < 25:
        return True
    for n in (8, 16, 32):
        grams = [text[i:i + n] for i in range(0, max(1, len(text) - n), n)]
        if len(grams) >= 3 and len(set(grams)) / len(grams) < 0.3:
            return True
    return False


LITERAL_RE = re.compile(r"「([^」]{1,40})」")

# PE 自带内容审查，拒绝时会返回一句说明（有时是 {"error": "...", "helpful": false} 这种 JSON）。
# 这不能被当成"增强结果"塞回输入框——那样等于把提示词换成了错误信息。
REFUSAL_RE = re.compile(
    r"(cannot fulfill|can'?t fulfill|can'?t help with|i'?m sorry|i am sorry|"
    r"unable to (?:help|assist|comply)|not able to (?:help|assist)|"
    r"无法(?:满足|协助|提供|完成)|抱歉[，,]?\s*我|违反.{0,6}(?:政策|准则))",
    re.IGNORECASE)


def detect_refusal(text: str) -> str | None:
    """判断 PE 输出是不是拒绝/审查说明，返回说明文本；否则 None。"""
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            if obj.get("helpful") is False or "error" in obj:
                msg = obj.get("error") or obj.get("message") or "内容策略拒绝"
                return str(msg)[:400]
            return None            # 是正常的结构化输出（如 PE-I2I 的 output 字段）
    # 纯文本拒绝：短句 + 命中拒绝措辞
    if len(s) <= 600 and REFUSAL_RE.search(s):
        return s[:400]
    return None


def lost_literal(original: str, enhanced: str) -> str | None:
    """检查原文「」里的文字有没有被 PE 翻译/改掉。返回丢失的第一个片段。"""
    for frag in LITERAL_RE.findall(original):
        if frag not in enhanced:
            return frag
    return None


def unwrap_pe_json(text: str) -> str:
    """PE-I2I 原生输出是 {"output": "..."}，取出里面的自然语言。"""
    s = text.strip()
    if not s.startswith("{"):
        return text
    try:
        obj = json.loads(s)
        for key in ("output", "text", "prompt", "instruction", "description", "edited_prompt"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return text
    except Exception:
        # JSON 被 max_length 截断时，用正则把第一个字符串字段抠出来
        m = re.search(r'"(?:output|text|prompt|instruction|description)"\s*:\s*("(?:[^"\\]|\\.)*")', s)
        if m:
            try:
                return json.loads(m.group(1)).strip()
            except Exception:
                pass
        return text


def wait_text(pid: str, timeout=240) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.5)
        entry = http_json(COMFY + "/history/%s" % pid, timeout=10).get(pid)
        if not entry:
            continue
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            raise RuntimeError(_error_text(status.get("messages") or []))
        outs = entry.get("outputs") or {}
        for node in outs.values():
            txt = node.get("text")
            if txt:
                value = txt[0] if isinstance(txt, list) else txt
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if status.get("completed"):
            raise RuntimeError("增强结束但没有返回文本")
    raise RuntimeError("增强超时")


def run_job(job_id: str, opts: dict) -> None:
    job = JOBS[job_id]
    cancel = job["cancel"]
    seeds = opts.get("seeds") or []
    try:
        for idx, seed in enumerate(seeds):
            if cancel.is_set():                   # 上一张刚出完就点了停止
                job["cancelled"] = True
                break
            job["current"] = idx + 1
            graph = build_graph(opts.get("engine") or STATE["engine"],
                                opts["prompt"], opts["width"], opts["height"],
                                opts["steps"], seed, opts["clip"], opts["refs"],
                                OUTPUT_PREFIX, opts.get("resolution", 1024))
            pid = submit_one(graph)
            job["prompt_id"] = pid
            if cancel.is_set():                   # 提交和停止撞车，补一刀
                comfy_interrupt()
            out = wait_one(pid, cancel=cancel)
            out["seed"] = seed
            with LOCK:
                job["images"].append(out)
    except Cancelled:
        job["cancelled"] = True
        log("任务 %s 已停止（已出 %d/%d 张）"
            % (job_id, len(job["images"]), job.get("total", 1)))
    except Exception as exc:                      # noqa: BLE001
        job["error"] = _friendly(exc)
    finally:
        # 已经出好的图照存历史：中途停止不该把前面几张一起丢掉
        if job["images"]:
            try:
                add_history({
                    "mode": opts.get("mode", "t2i"),
                    "engine": opts.get("engine") or STATE["engine"],
                    "prompt": opts["prompt"],
                    "steps": opts["steps"],
                    "batch": job.get("total", 1),
                    "clip": opts["clip"],
                    "size": "%dx%d" % (opts["width"], opts["height"]),
                    "resolution": opts.get("resolution", 1024),
                    "seed": seeds[0] if seeds else None,
                    "refs": opts.get("refs", []),
                    "file": job["images"][0]["filename"],
                    # 供前端估算批量耗时用
                    "elapsed": round(time.time() - job["start"], 1),
                })
            except Exception as exc:              # noqa: BLE001
                log("写历史失败：%s" % exc)
        job["elapsed"] = time.time() - job["start"]
        job["done"] = True


def _error_text(messages) -> str:
    for m in messages:
        try:
            if m[0] == "execution_error":
                return str(m[1].get("exception_message") or m[1])[:600]
        except Exception:
            continue
    return "执行出错"


def _friendly(exc: Exception) -> str:
    text, low = str(exc), str(exc).lower()
    if "out of memory" in low or "allocation" in low:
        return "显存不足。先关闭游戏/壁纸/模拟器，或把尺寸降到 896 以下、批量减到 1。"
    if "not in list" in low or "value not in" in low:
        return "模型文件缺失或名称不匹配：" + text[:300]
    return text[:600]


# ------------------------------------------------------------------ 窗口存活

def window_alive(title: str) -> bool:
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        found = []
        CB = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _cb(hwnd, _lp):
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                if n > 0:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    if title.lower() in buf.value.lower():
                        found.append(hwnd)
            return True

        user32.EnumWindows(CB(_cb), 0)
        return bool(found)
    except Exception:
        return True


def watchdog(title: str) -> None:
    """窗口关闭后退出，但必须先确认前端确实没心跳了。

    只用窗口标题判断会误杀：Edge 在某些状态下（切标签、全屏、失焦重建窗口）
    标题枚举可能漏掉，实测导致过"服务凭空消失、页面 Failed to fetch"。
    因此要求「窗口连续 3 次找不到」且「心跳已停 30 秒」两个条件同时成立。
    """
    time.sleep(5)
    missing = 0
    while not SHUTDOWN.is_set():
        time.sleep(2)
        gone = False
        if title:
            if window_alive(title):
                missing = 0
            else:
                missing += 1
                gone = missing >= 3
        # 只有"窗口真的不在了"才退出。
        # 不能只看心跳：窗口最小化/被遮挡/锁屏时 Chromium 会冻结定时器，
        # 心跳会停，但窗口和程序都还活着——早期版本就是这么误杀自己的。
        if gone and time.time() - LAST_BEAT > 15:
            log("窗口已关闭且无心跳，退出")
            request_shutdown()
            return


SHUTDOWN = threading.Event()


def request_shutdown():
    SHUTDOWN.set()


# ------------------------------------------------------------------ 图库

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _output_path(name: str) -> str | None:
    """把文件名解析成 output 目录内的绝对路径；越界或非法返回 None。"""
    root = os.path.abspath(COMFY_OUT)
    p = os.path.abspath(os.path.join(root, safe_name(name)))
    if not p.startswith(root + os.sep):
        return None
    return p


def list_outputs(limit: int = 60, offset: int = 0, query: str = "") -> dict:
    root = os.path.abspath(COMFY_OUT)
    try:
        names = os.listdir(root)
    except OSError:
        return {"total": 0, "items": []}
    q = query.lower()
    files = []
    for n in names:
        if not n.lower().endswith(IMAGE_EXTS):
            continue
        if q and q not in n.lower():
            continue
        p = os.path.join(root, n)
        try:
            if not os.path.isfile(p):
                continue
            st = os.stat(p)
        except OSError:
            continue
        files.append({"name": n, "size": st.st_size, "mtime": int(st.st_mtime)})
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return {"total": len(files), "items": files[offset:offset + limit]}


# ------------------------------------------------------------------ 历史记录

def load_history() -> list:
    """读历史。顺手给老数据补 id（删单条需要稳定标识）。"""
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    for i, it in enumerate(data):
        if isinstance(it, dict) and not it.get("id"):
            it["id"] = "%s-%d" % (it.get("t", 0), i)
    return data


def save_history(items: list) -> None:
    try:
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items[:HISTORY_MAX], f, ensure_ascii=False, indent=1)
        os.replace(tmp, HISTORY_FILE)
    except Exception as exc:                 # noqa: BLE001
        log("写历史失败: %s" % exc)


def add_history(entry: dict) -> list:
    with LOCK:
        items = load_history()
        entry = dict(entry or {})
        entry["t"] = int(entry.get("t") or time.time())
        entry["id"] = uuid.uuid4().hex[:12]
        items.insert(0, entry)
        # 简单去重：同提示词+同参数的连续重复不再叠加
        seen, uniq = set(), []
        for it in items:
            key = (it.get("prompt"), it.get("mode"), it.get("seed"), it.get("file"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(it)
        save_history(uniq)
        return uniq[:HISTORY_MAX]


def delete_history(ids: list) -> list:
    """按 id 删除历史条目，返回剩余列表。"""
    want = {str(i) for i in (ids or [])}
    with LOCK:
        items = load_history()
        kept = [it for it in items if str(it.get("id")) not in want]
        save_history(kept)
        return kept[:HISTORY_MAX]


def clear_history() -> None:
    with LOCK:
        save_history([])


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "ImageLab/1.0"
    # 图库一次要拉几十张缩略图，HTTP/1.0 每次都新建连接会拖垮加载
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    # ---------- GET ----------
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path, qs = u.path, urllib.parse.parse_qs(u.query)

        if path in ("/", "/index.html"):
            try:
                with open(UI_HTML, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(500, b"ui.html missing", "text/plain")

        if path == "/api/library":
            limit = max(1, min(200, int((qs.get("limit") or ["60"])[0])))
            offset = max(0, int((qs.get("offset") or ["0"])[0]))
            query = (qs.get("q") or [""])[0].strip()
            return self._json(200, list_outputs(limit, offset, query))

        if path == "/api/history":
            return self._json(200, {"items": load_history()})

        if path == "/api/health":
            v = vram_mib()
            return self._json(200, {
                "ready": STATE["ready"], "error": STATE["error"],
                "app": APP_NAME,
                "engine": engine_info(),
                "engines": list_engines(),
                "models": STATE["models"], "encoders": STATE["encoders"],
                "vram": {"used": v[0], "total": v[1]} if v else None})

        if path == "/api/progress":
            jid = (qs.get("id") or [""])[0]
            job = JOBS.get(jid)
            if not job:
                return self._json(404, {"error": "unknown job"})
            return self._json(200, {
                "done": bool(job.get("done")),
                "cancelled": bool(job.get("cancelled")),
                "images": job.get("images", []),
                "total": job.get("total", 1),
                "current": job.get("current", 0),
                "elapsed": round(job.get("elapsed") or (time.time() - job["start"]), 1),
                "peak_mib": job.get("peak"),
                "error": job.get("error")})

        if path == "/api/view":
            fn = (qs.get("filename") or [""])[0]
            sub = (qs.get("subfolder") or [""])[0]
            # 参考图在 input 目录，出图在 output 目录
            kind = (qs.get("type") or ["output"])[0]
            if kind not in ("output", "input", "temp"):
                kind = "output"

            # 优先直接读磁盘：图库一次要几十张缩略图，走 ComfyUI 代理并发高时会被丢
            base = {"output": COMFY_OUT, "input": COMFY_IN,
                    "temp": os.path.join(COMFY_ROOT, "temp")}.get(kind, COMFY_OUT)
            root = os.path.abspath(base)
            target = os.path.abspath(os.path.join(root, safe_name(fn)))
            if sub:
                target = os.path.abspath(os.path.join(
                    root, safe_name(sub), safe_name(fn)))
            if target.startswith(root + os.sep) and os.path.isfile(target):
                ext = os.path.splitext(target)[1].lower()
                ctype = {".png": "image/png", ".jpg": "image/jpeg",
                         ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "application/octet-stream")
                try:
                    with open(target, "rb") as f:
                        return self._send(200, f.read(), ctype)
                except OSError:
                    pass

            # 兜底：交给 ComfyUI 解析（子目录、临时文件等情况）
            url = COMFY + "/view?" + urllib.parse.urlencode(
                {"filename": fn, "subfolder": sub, "type": kind})
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
                    ctype = r.headers.get("Content-Type") or "image/png"
                    return self._send(200, data, ctype)
            except Exception as exc:             # noqa: BLE001
                return self._send(404, str(exc).encode(), "text/plain")

        if path == "/api/beat":
            global LAST_BEAT
            LAST_BEAT = time.time()
            return self._json(200, {"ok": True})

        self._json(404, {"error": "not found"})

    # ---------- POST ----------
    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/upload":
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return self._json(400, {"error": "空文件"})
            raw = self.rfile.read(n)
            name = safe_name(self.headers.get("X-Filename") or "ref.png")
            root, ext = os.path.splitext(name)
            if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                ext = ".png"
            final = "%s_%d%s" % (root[:60], int(time.time()), ext)
            os.makedirs(COMFY_IN, exist_ok=True)
            with open(os.path.join(COMFY_IN, final), "wb") as f:
                f.write(raw)
            return self._json(200, {"name": final})

        try:
            body = self._body_json()
        except Exception:
            return self._json(400, {"error": "bad json"})

        if path == "/api/engine":
            eid = body.get("id") or ""
            info = resolve_engine(eid)
            if not info:
                return self._json(400, {"error": "未知引擎：%s" % eid})
            if not info["installed"]:
                return self._json(400, {"error": "「%s」的权重未安装" % info["name"]})
            STATE["engine"] = eid
            log("切换引擎 -> %s" % info["name"])
            return self._json(200, {"engine": info, "engines": list_engines()})

        if path == "/api/history/add":
            entry = body.get("entry") if isinstance(body.get("entry"), dict) else body
            if not entry or not entry.get("prompt"):
                return self._json(400, {"error": "缺少提示词"})
            items = add_history(entry)
            return self._json(200, {"items": items})

        if path == "/api/history/import":
            incoming = body.get("items") or []
            if not isinstance(incoming, list):
                return self._json(400, {"error": "items 必须是数组"})
            existing = load_history()
            if existing:
                return self._json(200, {"items": existing, "imported": 0})
            cleaned = []
            for it in incoming[:HISTORY_MAX]:
                if isinstance(it, dict) and it.get("prompt"):
                    cleaned.append(it)
            save_history(cleaned)
            log("导入历史 %d 条" % len(cleaned))
            return self._json(200, {"items": cleaned, "imported": len(cleaned)})

        if path == "/api/history/delete":
            ids = body.get("ids") or ([body["id"]] if body.get("id") else [])
            if not ids:
                return self._json(400, {"error": "缺少 id"})
            items = delete_history(ids)
            log("删除历史 %d 条" % len(ids))
            return self._json(200, {"items": items, "removed": len(ids)})

        if path == "/api/history/clear":
            clear_history()
            log("清空历史")
            return self._json(200, {"items": []})

        if path == "/api/library/delete":
            names = body.get("names") or []
            removed, failed = [], []
            for n in names:
                p = _output_path(n)
                if not p or not os.path.exists(p):
                    failed.append(n)
                    continue
                try:
                    os.remove(p)
                    removed.append(n)
                except Exception:                # noqa: BLE001
                    failed.append(n)
            if removed:
                log("图库删除 %d 张" % len(removed))
            return self._json(200, {"removed": removed, "failed": failed})

        if path == "/api/open-folder":
            name = body.get("name") or ""
            p = _output_path(name) if name else None
            try:
                if p and os.path.exists(p):
                    subprocess.Popen(["explorer.exe", "/select,", p],
                                     creationflags=CREATE_NO_WINDOW)
                else:
                    subprocess.Popen(["explorer.exe", os.path.abspath(COMFY_OUT)],
                                     creationflags=CREATE_NO_WINDOW)
            except Exception as exc:             # noqa: BLE001
                return self._json(500, {"error": str(exc)})
            return self._json(200, {"ok": True})

        if path == "/api/ref-from-output":
            fn = safe_name(body.get("filename") or "")
            src = os.path.join(COMFY_OUT, fn)
            if not os.path.exists(src):
                return self._json(404, {"error": "找不到该输出文件"})
            final = "ref_%d_%s" % (int(time.time()), fn)
            os.makedirs(COMFY_IN, exist_ok=True)
            with open(src, "rb") as fi, open(os.path.join(COMFY_IN, final), "wb") as fo:
                fo.write(fi.read())
            return self._json(200, {"name": final})

        if path == "/api/enhance":
            if not STATE["ready"]:
                return self._json(503, {"error": STATE["error"] or "后端还没就绪"})
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                return self._json(400, {"error": "先写点提示词再增强"})
            kind = "i2i" if body.get("mode") == "edit" else "t2i"
            pe = STATE["models"].get("pe_" + kind) or STATE["models"].get("pe_t2i")
            if not pe:
                return self._json(400, {"error": "PE 增强模型未安装"})
            refs = [safe_name(r) for r in (body.get("refs") or [])][:1]
            t0 = time.time()
            try:
                pid = submit_one(build_pe_graph(prompt, kind, refs))
                text = wait_text(pid)
            except Exception as exc:             # noqa: BLE001
                return self._json(500, {"error": _friendly(exc)})

            # PE 自带内容审查：拒绝时产出的是一段"我不能…"，必须当错误处理，
            # 否则会被当成增强结果覆盖掉用户原文。
            refusal = detect_refusal(text)
            if refusal:
                log("PE 拒绝：%s" % refusal[:120])
                return self._json(422, {
                    "refused": True,
                    "error": "提示词增强被模型的安全策略拒绝了：%s" % refusal[:160],
                    "detail": refusal,
                })

            if looks_broken(text):
                return self._json(502, {"error": "增强结果异常（重复或过短），已保留你的原文。再点一次试试。"})
            text = unwrap_pe_json(text)
            gone = lost_literal(prompt, text)
            if gone:
                return self._json(502, {"error": "增强把「%s」改写掉了，已保留你的原文。再点一次，或手动把文字补回去。" % gone})
            return self._json(200, {"text": text, "model": pe,
                                    "elapsed": round(time.time() - t0, 1)})

        if path == "/api/generate":
            if not STATE["ready"]:
                return self._json(503, {"error": STATE["error"] or "后端还没就绪"})
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                return self._json(400, {"error": "提示词不能为空"})
            refs = [safe_name(r) for r in (body.get("refs") or [])][:10]
            if body.get("mode") == "edit" and not refs:
                return self._json(400, {"error": "编辑模式需要至少一张参考图"})

            batch = max(1, min(32, int(body.get("batch") or 1)))
            seed = int(body.get("seed") or 0)
            if seed:
                seeds = [(seed + i) % (2 ** 31) for i in range(batch)]
            else:
                seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(batch)]

            encs = STATE["encoders"]
            clip = body.get("clip") or STATE["models"]["clip"]
            if encs and clip not in encs:
                clip = encs[0]

            eid = body.get("engine") or STATE["engine"]
            einfo = resolve_engine(eid)
            if not einfo:
                return self._json(400, {"error": "未知引擎：%s" % eid})
            if not einfo["installed"]:
                return self._json(400, {"error": "「%s」的权重未安装" % einfo["name"]})
            STATE["engine"] = eid

            jid = "%d%04d" % (int(time.time()), random.randint(0, 9999))
            JOBS[jid] = {"start": time.time(), "done": False, "peak": 0,
                         "images": [], "total": batch, "current": 0,
                         "cancel": threading.Event(), "cancelled": False}
            threading.Thread(target=run_job, daemon=True, args=(jid, {
                "engine": eid,
                "prompt": prompt, "width": int(body.get("width") or 1024),
                "height": int(body.get("height") or 1024),
                "steps": int(body.get("steps") or 20), "seeds": seeds,
                "clip": clip, "refs": refs,
                "mode": body.get("mode") or "t2i",
                "resolution": int(body.get("resolution") or 1024)})).start()
            threading.Thread(target=watch_vram, args=(JOBS[jid],), daemon=True).start()
            return self._json(200, {"id": jid, "seeds": seeds})

        if path == "/api/cancel":
            jid = (body.get("id") or "").strip()
            job = JOBS.get(jid)
            if not job:
                return self._json(404, {"error": "任务不存在或已结束"})
            if job.get("done"):
                return self._json(200, {"ok": True, "done": True})
            job.setdefault("cancel", threading.Event()).set()
            stopped = comfy_interrupt()
            log("用户停止任务 %s（ComfyUI 中断%s）"
                % (jid, "已发出" if stopped else "没发出去"))
            return self._json(200, {"ok": True, "interrupted": stopped})

        if path == "/api/quit":
            self._json(200, {"ok": True})
            threading.Timer(0.3, request_shutdown).start()
            return

        self._json(404, {"error": "not found"})


def port_free(port: int) -> bool:
    """检测端口是否可绑定。

    Windows 上 SO_REUSEADDR 语义和 Unix 不同：它允许绑定已在监听的端口，
    这会让检测误判为"空闲"，结果真正 listen 时抛 WSAEADDRINUSE 直接退出。
    所以 Windows 下不设该选项。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name != "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((UI_HOST, port))
            return True
        except OSError:
            return False


def launch_window(url: str) -> None:
    for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(exe):
            subprocess.Popen(
                [exe, "--app=" + url, "--window-size=1260,900",
                 "--user-data-dir=" + os.path.join(APP_DIR, "_window_profile"),
                 "--no-first-run", "--no-default-browser-check",
                 "--disable-features=Translate"],
                creationflags=CREATE_NO_WINDOW)
            return
    import webbrowser
    webbrowser.open(url)


def main() -> int:
    global UI_PORT, LAST_BEAT
    no_window = "--no-window" in sys.argv

    scan_models()
    # 给老历史补出 id 并落盘：删除单条需要稳定标识，不能每次读都重新编
    try:
        items = load_history()
        if items:
            save_history(items)
    except Exception:
        pass
    threading.Thread(target=boot_comfy, daemon=True).start()

    while not port_free(UI_PORT):
        UI_PORT += 1
    srv = None
    for _ in range(20):                       # 端口被抢时继续往后找，别直接退出
        try:
            srv = ThreadingHTTPServer((UI_HOST, UI_PORT), Handler)
            break
        except OSError:
            log("端口 %d 绑定失败，换下一个" % UI_PORT)
            UI_PORT += 1
    if srv is None:
        log("找不到可用端口，退出")
        return 1
    url = "http://%s:%d/" % (UI_HOST, UI_PORT)
    log("界面地址 %s" % url)

    LAST_BEAT = time.time()
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    if not no_window:
        launch_window(url)
        threading.Thread(target=watchdog, args=(APP_NAME,), daemon=True).start()

    log("运行中")
    try:
        while not SHUTDOWN.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    log("正在退出")
    srv.shutdown()
    shutdown_comfy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
