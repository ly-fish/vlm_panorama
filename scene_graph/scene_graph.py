import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pathlib import Path

# 加载环境变量
current_dir = Path(__file__).resolve().parent
env_path = current_dir /".env"
load_dotenv(dotenv_path=env_path)

class SceneGraph:
    """
    基于 SE360 框架逻辑构建的教学场景图生成器。
    该类整合了物理附属关系分析 (PAA) 与空间位置解耦推理。
    """
    
    def __init__(self):
        # 从环境变量获取配置
        api_key = os.getenv("QWEN_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL")
        
        if not api_key:
            raise ValueError("未在环境变量中找到 QWEN_API_KEY，请检查 .env 文件。")

        # 初始化大语言模型，采用低采样温度以保证结构化输出的稳定性
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model="qwen-max",
            temperature=0.1 
        )
        self.parser = JsonOutputParser()

    def generate(self, user_instruction: str) -> Dict[str, Any]:
        """
        解析用户指令，生成符合全景几何逻辑的教学场景图。
        """
        
        # 构建系统提示词，嵌入 SE360 论文的核心技术逻辑
        system_prompt = """
        You are an AI collaborator specializing in Pedagogical Scene Design for 360° virtual environments.
        Your task is to transform user intent into a 'Pedagogical Scene Graph' based on the SE360 methodology[cite: 9, 46].

        ### Core Reasoning Requirements:
        1. **Physical Affiliation Analysis (PAA)**: For every major object, identify items physically supported by it (e.g., items 'resting on', 'inside', or 'attached to' the host) to ensure object completeness[cite: 98, 625].
        2. **Spatial Localization Classification**: Determine if a description uses 'absolute' positioning (relative to the 360° frame/room structure) or 'relative' positioning (relative to other objects)[cite: 118, 705].
        3. **Spherical Consistency**: Recommend 'position_hints' that avoid placing critical instructional items at the longitudinal seams unless specified[cite: 23, 71].
        4. **Hierarchical Descriptions**: Provide both a detailed 'Standard Refined Description' and a 'Brief Description' for each object[cite: 115, 116].

        ### JSON Output Format:
        {{
          "scene_theme": "Theme of the environment",
          "subject_domain": "e.g., Chemistry, Physics",
          "pedagogical_goal": "Educational purpose of this scene",
          "objects": [
            {{
              "id": "obj_unique_id",
              "name": "Specific object name",
              "category": "Precise category [cite: 571]",
              "importance": "high/medium/low",
              "required": true,
              "physical_parent": "ID of the supporting object if applicable [cite: 146]",
              "attributes": {{
                "visibility": "clear/ambient",
                "position_hint": "Location context"
              }}
            }}
          ],
          "spatial_relations": [
            {{ 
              "subject": "obj_id", 
              "type": "absolute/relative [cite: 902]", 
              "relation": "e.g., next_to, on_top_of", 
              "target": "target_id or frame_location" 
            }}
          ],
          "safety_constraints": ["Safety rules derived from physical context [cite: 51]"],
          "success_criteria": ["Metrics for visual and pedagogical accuracy [cite: 365]"]
        }}

        Return ONLY the JSON object.
        """

        # 创建 LangChain 提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Instruction: {instruction}")
        ])

        # 构建处理链
        chain = prompt | self.llm | self.parser
        
        try:
            return chain.invoke({"instruction": user_instruction})
        except Exception as e:
            return {{"error": f"Failed to generate scene graph: {str(e)}"}}

# --- 模块化调用示例 ---
if __name__ == "__main__":
    # 初始化 SceneGraph
    sg = SceneGraph()

    # 用户输入的教学指令
    instruction = (
        "Create a high school chemistry lab for titration. "
        "The center should feature a heavy lab table with a burette and stand on it. "
        "A safety poster must be near the exit, and dangerous acids should be "
        "locked in a cabinet far from the student benches."
    )

    # 生成场景图
    result = sg.generate(instruction)
    
    # 打印结果
    print(json.dumps(result, indent=2))