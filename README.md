# SupportPilot — AI Support Agent

A Dockerised customer-support AI system that combines a LangGraph agent with FastAPI, Streamlit, PostgreSQL, Redis, Keycloak, and a FastMCP tool server to handle support tickets end-to-end. The agent reads business data through MCP tools, writes through local LangChain tools, and persists conversation memory in Redis with a 7-day TTL.

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Agent & Tools](#agent--tools)
- [MCP Server](#mcp-server)
- [Development](#development)
- [Testing](#testing)
