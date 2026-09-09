"""Safe, allowlisted tools the assistant can execute.

Each tool is registered via the @tool decorator in tools/registry.py.
Modules here import their own registry tools by name so future tools
(e.g. whatsapp) can be added as a new module with zero changes elsewhere.
"""
