#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_image_gen.py
调用阿里云百炼(DashScope) Qwen 图像生成模型 (qwen-image-3.0-pro) 生成图片并保存为 PNG。

用法:
    python qwen_image_gen.py                             # 默认提示词, 输出 qwen_image.png
    python qwen_image_gen.py "你的提示词"                 # 自定义提示词
    python qwen_image_gen.py "提示词" "输出文件.png"      # 自定义提示词 + 文件名

依赖: 仅 Python 标准库, 无需 pip 安装任何第三方包。

API Key 取值优先级:
    1. 环境变量 DASHSCOPE_API_KEY
    2. 下方 DEFAULT_API_KEY 常量
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL = "qwen-image-3.0-pro"
BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# 环境变量 DASHSCOPE_API_KEY 未设置时使用的默认 Key(建议改用环境变量)
DEFAULT_API_KEY = "请配置"

DEFAULT_OUT = "qwen_image.png"

DEFAULT_PROMPT = (
    "画面是一张竖幅户外人像摄影，整体从上到下呈现温暖的午后街景氛围。"
    "顶部左侧到上方大面积被深绿色藤蔓和橙色小花覆盖，花叶从建筑檐口自然垂落，"
    "受阳光照射的叶片呈黄绿色高光，阴影处则偏深绿，形成浓密而柔和的背景层次。"
    "左上至中上区域是一块深蓝色横向招牌，招牌表面较暗、略带磨砂质感，"
    "上面以白色哥特体大字写着 Il Messaggero，文字位于画面左侧偏上，部分被前景花叶轻微遮挡，"
    "字体高对比、带装饰性尖角和粗细变化。"
    "招牌下方是报刊亭或书报摊的玻璃展示窗，黑色金属框架将橱窗分隔成多个矩形区域，"
    "内部陈列着许多报纸、杂志和书刊封面，但大多因景深虚化和光线反射而难以辨读，"
    "形成浅色纸张与深色边框交错的背景纹理。"
    "画面右上方是强烈的逆光区域，阳光从街道尽头照入，背景建筑被虚化成米灰色块面，"
    "边缘柔和，呈现明显的浅景深效果。"
    "画面中部偏右是一名年轻成年女性的半身至膝上人像，她回头面向镜头微笑，"
    "身体略向右转，肩背朝向观者，姿态自然放松。"
    "她有长而浓密的黑色波浪卷发，发丝被逆光勾勒出金色轮廓光，发梢在右侧向外散开，"
    "显得轻盈蓬松。她肤色白皙，脸型柔和偏鹅蛋形，眉形细致，眼睛明亮，眼妆清透，"
    "睫毛明显，面部带有自然高光，唇部为柔和珊瑚红色，笑容露齿，表情亲切明朗。"
    "她佩戴小巧耳饰，身穿黑色细肩带露背连衣裙，面料颜色深黑、轮廓简洁，"
    "细肩带从肩部向背部延伸，背部线条清晰。"
    "画面下部偏左到中部，她双手抱着一束玫瑰花，花束体积较大，"
    "主要由橙色、杏色、粉色和浅桃色玫瑰组成，花瓣层层卷曲，边缘被阳光照亮，"
    "绿色叶片和长花茎从花束下方垂出，花束与黑色裙装形成鲜明色彩对比。"
    "右侧背景是一条被阳光照亮的城市街道，地面呈暖灰与金黄色调，"
    "远处建筑、街边设施和一个模糊的红色圆形交通标志位于右下远景，均因焦外虚化而只保留色块和轮廓。"
    "整张照片采用暖色胶片感处理，带有细腻颗粒、柔和对比和明显逆光边缘光，人物位于视觉焦点，"
    "背景报刊亭、花藤、街道和阳光共同营造出浪漫、明亮、都市漫步式的氛围。"
)


def _fix_console_encoding():
    """避免 Windows 控制台(GBK)输出中文时报 UnicodeEncodeError。"""
    if os.name != "nt":
        return
    try:
        os.system("chcp 65001 >nul")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_api_key():
    return os.environ.get("DASHSCOPE_API_KEY") or DEFAULT_API_KEY


def request_image(prompt, api_key):
    """提交生成请求, 返回图片 URL 或 base64 字符串(及其类型)。"""
    body = {
        "model": MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": {"prompt_extend": True},
    }

    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )

    print("[1/3] 提交生成任务 ...")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        print("!! 接口返回错误 (HTTP %s):" % e.code)
        print(detail)
        sys.exit(1)
    except urllib.error.URLError as e:
        print("!! 网络请求失败:", e.reason)
        sys.exit(1)

    return parse_image_data(data)


def parse_image_data(data):
    """从响应中提取图片。返回 (kind, value), kind 为 'url' 或 'b64'。"""
    # 标准同步返回: output.choices[0].message.content[0].image (URL)
    try:
        url = data["output"]["choices"][0]["message"]["content"][0]["image"]
        if url:
            return "url", url
    except (KeyError, IndexError, TypeError):
        pass

    # 兼容其它返回结构
    try:
        url = data["output"]["results"][0]["url"]
        if url:
            return "url", url
    except (KeyError, IndexError, TypeError):
        pass

    try:
        b64 = data["output"]["choices"][0]["message"]["content"][0]["b64_json"]
        if b64:
            return "b64", b64
    except (KeyError, IndexError, TypeError):
        pass

    print("!! 未在响应中找到图片数据, 完整响应如下:")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    sys.exit(1)


def download(url, out_path):
    print("[2/3] 下载图片 ...")
    print("图片 URL: %s" % url)
    with urllib.request.urlopen(url, timeout=300) as resp:
        content = resp.read()
    with open(out_path, "wb") as f:
        f.write(content)
    return len(content)


def main():
    _fix_console_encoding()

    parser = argparse.ArgumentParser(
        description="调用 DashScope Qwen 图像生成模型 (%s) 生成图片" % MODEL
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="提示词, 缺省使用内置默认提示词")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUT, help="输出 PNG 文件名(可含路径), 默认 qwen_image.png")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("!! 未找到 API Key, 请设置环境变量 DASHSCOPE_API_KEY 或在脚本内配置 DEFAULT_API_KEY")
        sys.exit(1)

    print("=" * 44)
    print(" Qwen 图像生成  (%s)" % MODEL)
    print("=" * 44)
    print("输出: %s" % args.output)
    print("提示词长度: %d 字符" % len(args.prompt))
    print()

    kind, value = request_image(args.prompt, api_key)

    if kind == "url":
        size = download(value, args.output)
    else:  # b64
        import base64
        print("[2/3] 解码 base64 图片 ...")
        raw = base64.b64decode(value)
        with open(args.output, "wb") as f:
            f.write(raw)
        size = len(raw)

    abs_path = os.path.abspath(args.output)
    print("[3/3] 完成 ✓")
    print("已保存: %s  (%d 字节)" % (abs_path, size))


if __name__ == "__main__":
    main()
