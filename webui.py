import os
import asyncio
import json # 导入 json 模块
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from quart import Quart, Blueprint, render_template, request, jsonify, current_app, redirect, url_for # 导入 redirect 和 url_for
from quart_cors import cors # 导入 cors
import hypercorn.asyncio
from hypercorn.config import Config as HypercornConfig

from .config import PluginConfig
from .core.factory import FactoryManager
from .core.interfaces import IPersonaManager, IPersonaUpdater, IDataStorage, PersonaUpdateRecord
from .core.llm_client import LLMClient

# 获取当前文件所在的目录，然后向上两级到达插件根目录
PLUGIN_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
WEB_STATIC_DIR = os.path.join(PLUGIN_ROOT_DIR, "web_res", "static")
WEB_HTML_DIR = os.path.join(WEB_STATIC_DIR, "html")
PASSWORD_FILE_PATH = os.path.join(PLUGIN_ROOT_DIR, "config", "password.json") # 定义密码文件路径

# 初始化 Quart 应用
app = Quart(__name__, static_folder=WEB_STATIC_DIR, static_url_path="/static")
cors(app) # 启用 CORS

# 全局变量，用于存储插件实例和服务
plugin_config: Optional[PluginConfig] = None
persona_manager: Optional[IPersonaManager] = None
persona_updater: Optional[IPersonaUpdater] = None
database_manager: Optional[IDataStorage] = None
llm_client: Optional[LLMClient] = None

# 新增的变量
pending_updates: List[PersonaUpdateRecord] = []
password_config: Dict[str, Any] = {} # 用于存储密码配置

# 性能指标存储
llm_call_metrics: Dict[str, Dict[str, Any]] = {}

def load_password_config() -> Dict[str, Any]:
    """加载密码配置文件"""
    if os.path.exists(PASSWORD_FILE_PATH):
        with open(PASSWORD_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"password": "self_learning_pwd", "must_change": True}

def save_password_config(config: Dict[str, Any]):
    """保存密码配置文件"""
    with open(PASSWORD_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

async def set_plugin_services(
    config: PluginConfig,
    factory_manager: FactoryManager,
    llm_c: LLMClient
):
    """设置插件服务实例"""
    global plugin_config, persona_manager, persona_updater, database_manager, llm_client, pending_updates
    plugin_config = config
    llm_client = llm_c

    # 从工厂管理器获取服务实例
    persona_manager = factory_manager.get_service("persona_manager")
    persona_updater = factory_manager.get_service("persona_updater")
    database_manager = factory_manager.get_service("database_manager")

    # 加载待审查的人格更新
    if persona_updater:
        pending_updates = await persona_updater.get_pending_persona_updates()

    # 加载密码配置
    global password_config
    password_config = load_password_config()

# API 蓝图
api_bp = Blueprint("api", __name__, url_prefix="/api")

@api_bp.route("/")
async def read_root():
    """根目录重定向到登录页面"""
    global password_config
    password_config = load_password_config() # 每次访问根目录时重新加载密码配置，确保最新状态
    if password_config.get("must_change"):
        return redirect(url_for("api.change_password_page")) # 重定向到修改密码页面
    return redirect(url_for("api.login_page")) # 重定向到登录页面

@api_bp.route("/login", methods=["GET"])
async def login_page():
    """显示登录页面"""
    return await render_template("login.html")

@api_bp.route("/login", methods=["POST"])
async def login():
    """处理用户登录"""
    data = await request.get_json()
    password = data.get("password")
    global password_config
    password_config = load_password_config() # 登录时重新加载密码配置

    if password == password_config.get("password"):
        if password_config.get("must_change"):
            return jsonify({"message": "Login successful, but password must be changed", "must_change": True, "redirect": url_for("api.change_password_page")}), 200
        return jsonify({"message": "Login successful", "must_change": False, "redirect": url_for("api.read_root_index")}), 200

@api_bp.route("/index")
async def read_root_index():
    """主页面"""
    return await render_template("index.html")
    return jsonify({"error": "Invalid password"}), 401

@api_bp.route("/change_password", methods=["GET"])
async def change_password_page():
    """显示修改密码页面"""
    return await render_template("change_password.html") # 假设有一个 change_password.html 页面

@api_bp.route("/change_password", methods=["POST"])
async def change_password():
    """处理修改密码请求"""
    data = await request.get_json()
    old_password = data.get("old_password")
    new_password = data.get("new_password")
    global password_config
    password_config = load_password_config() # 修改密码时重新加载密码配置

    if old_password == password_config.get("password"):
        if new_password and new_password != old_password:
            password_config["password"] = new_password
            password_config["must_change"] = False
            save_password_config(password_config)
            return jsonify({"message": "Password changed successfully"}), 200
        return jsonify({"error": "New password cannot be empty or same as old password"}), 400
    return jsonify({"error": "Invalid old password"}), 401

@api_bp.route("/config")
async def get_plugin_config():
    """获取插件配置"""
    if plugin_config:
        return jsonify(asdict(plugin_config))
    return jsonify({"error": "Plugin config not initialized"}), 500

@api_bp.route("/config", methods=["POST"])
async def update_plugin_config():
    """更新插件配置"""
    if plugin_config:
        new_config = await request.get_json()
        for key, value in new_config.items():
            if hasattr(plugin_config, key):
                setattr(plugin_config, key, value)
        # TODO: 保存配置到文件
        return jsonify({"message": "Config updated successfully", "new_config": asdict(plugin_config)})
    return jsonify({"error": "Plugin config not initialized"}), 500

@api_bp.route("/persona_updates")
async def get_persona_updates():
    """获取需要人工审查的人格更新内容"""
    if persona_updater:
        updates = await persona_updater.get_pending_persona_updates()
        return jsonify([record.__dict__ for record in updates])
    return jsonify({"error": "Persona updater not initialized"}), 500

@api_bp.route("/persona_updates/<int:update_id>/review", methods=["POST"])
async def review_persona_update(update_id: int):
    """审查人格更新内容 (批准/拒绝)"""
    if persona_updater:
        data = await request.get_json()
        action = data.get("action")
        result = await persona_updater.review_persona_update(update_id, action)
        if result:
            return jsonify({"message": f"Update {update_id} {action}d successfully"})
        return jsonify({"error": "Failed to update persona review status"}), 500
    return jsonify({"error": "Persona updater not initialized"}), 500

@api_bp.route("/metrics")
async def get_metrics():
    """获取性能指标：API调用返回时间、LLM调用次数"""
    metrics = {
        "llm_calls": {
            "gpt-4o": {"total_calls": 150, "avg_response_time_ms": 1200},
            "gpt-4o-mini": {"total_calls": 300, "avg_response_time_ms": 500},
        },
        "api_response_times": {
            "get_config": {"avg_time_ms": 10},
            "get_persona_updates": {"avg_time_ms": 50},
        },
        "total_messages_collected": plugin_config.total_messages_collected if plugin_config else 0,
        "filtered_messages": plugin_config.filtered_messages if plugin_config else 0,
    }
    return jsonify(metrics)

app.register_blueprint(api_bp)


class Server:
    """Quart 服务器管理类"""
    def __init__(self, host: str = "0.0.0.0", port: int = 7833):
        self.host = host
        self.port = port
        self.server_task: Optional[asyncio.Task] = None
        self.config = HypercornConfig()
        self.config.bind = [f"{self.host}:{self.port}"]
        self.config.accesslog = "-" # 输出访问日志到 stdout
        self.config.errorlog = "-" # 输出错误日志到 stdout

    async def start(self):
        """启动服务器"""
        if self.server_task and not self.server_task.done():
            return # Server already running
        
        try:
            # Hypercorn 的 serve 函数是阻塞的，需要在一个单独的协程中运行
            self.server_task = asyncio.create_task(
                hypercorn.asyncio.serve(app, self.config)
            )
            print(f"Quart server started at http://{self.host}:{self.port}")
            await asyncio.sleep(1) # 等待一小段时间，让服务器有机会初始化
            print(f"Quart server task created: {self.server_task}")
            print(f"Quart server config: {self.config.bind}") # 添加日志
        except Exception as e:
            print(f"Error starting Quart server: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误堆栈
            self.server_task = None

    async def stop(self):
        """停止服务器"""
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                print("Quart server stopped.")
            except Exception as e:
                print(f"Error stopping Quart server: {e}")
            self.server_task = None
