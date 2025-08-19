<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Rabeel-Ashraf/Profile-photo-and-logos-/raw/main/download.png">
  <img alt="Orionix Agent" src="https://github.com/Rabeel-Ashraf/Profile-photo-and-logos-/raw/main/download.png" width="400" />
</picture>

<h3>AI Agents That Think, Adapt, and Protect</h3>

<p>
  <a href="https://www.orionix.io/" target="_blank">🌐 Website</a> •
  <a href="https://www.orionix.io/docs/quickstart/installation" target="_blank">⚡ Quick Start</a> •
  <a href="https://discord.gg/yourdiscordlink" target="_blank">💬 Discord</a> •
  <a href="https://www.orionix.io/docs/quickstart/examples" target="_blank">📖 Examples</a>
</p>

<p>
  <a href="https://pypi.org/project/orionix-agent/"><img alt="PyPI" src="https://img.shields.io/pypi/v/orionix-agent?color=blue"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue">
  <a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
  <a href="https://discord.gg/yourdiscordlink"><img alt="Discord" src="https://img.shields.io/discord/yourdiscordid?color=7289da&logo=discord&logoColor=white"></a>
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/yourusername/orionix-agent?style=social">
</p>

</div>

## 🎯 The Problem Every AI Developer Faces

You build an AI agent. It works great in testing. Then real users start talking to it and...

- ❌ It ignores your carefully crafted system prompts
- ❌ It hallucinates responses in critical moments
- ❌ It can't handle edge cases consistently
- ❌ Each conversation feels like a roll of the dice

**Sound familiar?** You're not alone. This is the #1 pain point for developers building production AI agents.

## ⚡ The Solution: Teach Principles, Not Scripts

Orionix Agent flips the script on AI agent development. Instead of hoping your LLM will follow instructions, **Orionix guarantees it**.

```python
# Example: Define a guideline for your agent
import orionix_agent.sdk as o

@o.tool
async def get_weather(context: o.ToolContext, city: str) -> o.ToolResult:
    return o.ToolResult(f"Sunny, 72°F in {city}")

async def main():
    async with o.Server() as server:
        agent = await server.create_agent(
            name="WeatherBot",
            description="Helpful weather assistant"
        )

        await agent.create_guideline(
            condition="User asks about weather",
            action="Get current weather and provide a friendly response with suggestions",
            tools=[get_weather]
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
