from __future__ import annotations


ZH = "zh"
EN = "en"


TRANSLATIONS = {
    ZH: {
        "nav_workbench": "设计与优化工作台",
        "nav_optimizer": "智能优化",
        "nav_data": "历史数据智能",
        "nav_gpu": "GPU管理",
        "nav_lithography": "投影光刻目标",
        "nav_export": "数据与导出",
        "nav_about": "版权声明",
        "language": "语言",
        "version": "版本",
        "status_ready": "就绪",
        "refresh_preview": "刷新预览",
        "validate": "检查参数",
        "start_hybrid": "启动混合优化",
        "scan_data": "扫描历史数据",
        "consent_required": "需要用户授权后才能读取历史数据。",
        "gpu_refresh": "刷新显卡列表",
        "gpu_none": "未检测到可用于计算的显卡，当前使用 CPU 后端。",
        "gpu_select": "选择参与计算的 GPU",
        "export_bundle": "导出工程与结果",
        "build_note": "开源交付内容为源码包，可由使用单位按自身环境移植和二次开发。",
        "copyright_title": "版权声明",
        "copyright_body": "本开源软件按 MIT License 发布，可用于研究、教学、内部工程验证、修改和再分发；请保留许可证声明。",
        "motto": "",
        "lithography_title": "角放大超透镜投影光刻优化目标",
        "data_consent": "我同意软件读取我选择目录中的历史计算数据，用于机器学习再优化。",
        "installer": "Windows 安装包",
        "desktop_shortcut": "桌面快捷方式",
        "start_menu": "开始菜单快捷方式",
        "uninstall_data_choice": "卸载时会询问是否删除用户数据。",
    },
    EN: {
        "nav_workbench": "Design & Optimization",
        "nav_optimizer": "Smart Optimization",
        "nav_data": "Historical Data AI",
        "nav_gpu": "GPU Manager",
        "nav_lithography": "Projection Lithography",
        "nav_export": "Data & Export",
        "nav_about": "Copyright",
        "language": "Language",
        "version": "Version",
        "status_ready": "Ready",
        "refresh_preview": "Refresh Preview",
        "validate": "Validate",
        "start_hybrid": "Start Hybrid Optimization",
        "scan_data": "Scan Historical Data",
        "consent_required": "User consent is required before reading historical data.",
        "gpu_refresh": "Refresh GPUs",
        "gpu_none": "No compute-capable GPU detected. CPU backend is active.",
        "gpu_select": "Select GPUs for computation",
        "export_bundle": "Export Project and Results",
        "build_note": "This open-source delivery is a source package that downstream organizations may port and extend.",
        "copyright_title": "Copyright",
        "copyright_body": "This open-source software is released under the MIT License for research, teaching, internal engineering validation, modification and redistribution. Preserve the license notice.",
        "motto": "",
        "lithography_title": "Angular-Magnification Metalens Projection Lithography Targets",
        "data_consent": "I consent to reading historical calculation data from the selected folder for machine-learning re-optimization.",
        "installer": "Windows Installer",
        "desktop_shortcut": "Desktop shortcut",
        "start_menu": "Start Menu shortcut",
        "uninstall_data_choice": "The uninstaller asks whether to delete user data.",
    },
}


def tr(language: str, key: str) -> str:
    table = TRANSLATIONS.get(language, TRANSLATIONS[ZH])
    return table.get(key, TRANSLATIONS[ZH].get(key, key))
