"""
Configuration endpoints for the RAG Chatbot API.
Provides configuration data for the ChatKit UI.
"""
from fastapi import APIRouter
from typing import Dict, Any
from pydantic import BaseModel

router = APIRouter()

class ChatKitConfig(BaseModel):
    """Configuration model for ChatKit UI"""
    theme: str = "light"
    maxTokens: int = 1000
    temperature: float = 0.7
    streamingEnabled: bool = True
    maxHistory: int = 50
    enableMarkdown: bool = True
    enableCodeHighlighting: bool = True
    enableSources: bool = True
    enableTextSelection: bool = True
    enableMobileDrawer: bool = True
    enableDesktopPanel: bool = True
    fontSize: str = "medium"
    fontFamily: str = "system-ui"


@router.get("/config/chatkit")
async def get_chatkit_config() -> Dict[str, Any]:
    """
    Get configuration for ChatKit UI.
    Returns configuration parameters that the UI components can use.
    """
    config = ChatKitConfig()
    return config.dict()


@router.get("/config/chatkit/ui")
async def get_ui_config() -> Dict[str, Any]:
    """
    Get UI-specific configuration for ChatKit.
    Returns UI appearance and behavior settings.
    """
    ui_config = {
        "theme": "light",
        "primaryColor": "#00C26A",  # Green theme
        "secondaryColor": "#333333",
        "borderRadius": "8px",
        "spacing": {
            "small": "8px",
            "medium": "16px",
            "large": "24px"
        },
        "typography": {
            "fontSize": "14px",
            "fontFamily": "system-ui, -apple-system, sans-serif"
        },
        "features": {
            "enableAnimations": True,
            "enableTransitions": True,
            "enableTooltips": True
        }
    }
    return ui_config