# SuperClaude Quick Reference
# Beast Mode Prompt
/sc:pm --ultrathink --all-mcp --delegate auto --task-manage --loop --iterations 10 --validate --scope system --persona-python-expert "complete neural network competition"  


## /sc Commands

| Category | Command | Description |
|----------|---------|-------------|
| **Planning** | `/sc:brainstorm` | Structured requirements discovery |
| | `/sc:design` | System architecture design |
| | `/sc:estimate` | Time/effort estimation |
| | `/sc:spec-panel` | Multi-expert specification review |
| **Development** | `/sc:implement` | Code implementation |
| | `/sc:build` | Build & compile projects |
| | `/sc:improve` | Code improvements |
| | `/sc:cleanup` | Refactoring & dead code removal |
| | `/sc:explain` | Code explanation |
| **Testing** | `/sc:test` | Test generation & execution |
| | `/sc:analyze` | Code analysis (quality, security, perf) |
| | `/sc:troubleshoot` | Debugging & issue resolution |
| | `/sc:reflect` | Retrospectives & task reflection |
| **Docs** | `/sc:document` | Documentation generation |
| | `/sc:help` | Show all commands |
| **Git** | `/sc:git` | Git operations with smart commits |
| **Project** | `/sc:pm` | Project management orchestration |
| | `/sc:task` | Task tracking & workflow |
| | `/sc:workflow` | Workflow automation from PRDs |
| **Research** | `/sc:research` | Deep autonomous web research |
| | `/sc:business-panel` | Multi-expert business analysis |
| **Session** | `/sc:load` | Load session context |
| | `/sc:save` | Save session state |
| **Utility** | `/sc:agent` | AI agent coordination |
| | `/sc:spawn` | Parallel task delegation |
| | `/sc:select-tool` | Intelligent tool selection |
| | `/sc:recommend` | Command recommendations |
| | `/sc:index` | Project indexing & knowledge base |
| | `/sc:index-repo` | Repository indexing (94% token reduction) |
| | `/sc:sc` | Show all available commands |

---

## Behavioral Modes

| Mode | Trigger | What It Does |
|------|---------|--------------|
| **Brainstorming** | `--brainstorm` | Asks right questions, collaborative discovery |
| **Business Panel** | `/sc:business-panel` | Multi-expert strategic analysis |
| **Deep Research** | `/sc:research` | Autonomous web research with multi-hop reasoning |
| **Orchestration** | `--orchestrate` | Efficient tool coordination & parallel execution |
| **Token-Efficiency** | `--uc` | 30-50% context savings via symbol system |
| **Task Management** | `--task-manage` | Systematic organization with memory |
| **Introspection** | `--introspect` | Meta-cognitive analysis & reasoning transparency |

**Usage:**
```bash
/sc:implement --brainstorm "new feature idea"
/sc:analyze --orchestrate src/
/sc:task --task-manage "refactor auth system"
/sc:explain --introspect "why did this fail"
```

---

## Research Depth Levels

| Depth | Sources | Hops | Time | Best For |
|:-----:|:-------:|:----:|:----:|----------|
| **Quick** | 5-10 | 1 | ~2min | Quick facts, simple queries |
| **Standard** | 10-20 | 3 | ~5min | General research (default) |
| **Deep** | 20-40 | 4 | ~8min | Comprehensive analysis |
| **Exhaustive** | 40+ | 5 | ~10min | Academic-level research |

**Usage:**
```bash
/sc:research "what is React hooks"                    # Quick - simple query
/sc:research "AI trends 2025"                         # Standard - general research
/sc:research --think-hard "microservices patterns"    # Deep - comprehensive
/sc:research --ultrathink "quantum computing state"   # Exhaustive - academic level
```

**Research Strategies:**
- **Planning-Only**: Direct execution for clear queries
- **Intent-Planning**: Clarification for ambiguous requests
- **Unified**: Collaborative plan refinement (default)

---

## Specialized Agents

Call via `--persona-[agent-name]` or auto-activated by context.

**Usage:**
```bash
--persona-python-expert
--persona-refactoring-expert
--persona-security-engineer
```

### Technical Specialists
| Agent | Focus |
|-------|-------|
| `python-expert` | Production Python, SOLID principles, security |
| `system-architect` | Scalable architecture, long-term decisions |
| `frontend-architect` | Accessible UIs, React/Vue/Angular, UX |
| `backend-architect` | APIs, fault tolerance, data integrity |
| `security-engineer` | Vulnerability detection, compliance |
| `performance-engineer` | Bottleneck elimination, optimization |

### Quality & Process
| Agent | Focus |
|-------|-------|
| `refactoring-expert` | Clean code, technical debt reduction |
| `quality-engineer` | Testing strategies, edge case detection |
| `devops-architect` | CI/CD, Docker, Kubernetes, reliability |
| `root-cause-analyst` | Evidence-based debugging, hypothesis testing |
| `requirements-analyst` | Requirement discovery from ambiguity |

### Knowledge & Guidance
| Agent | Focus |
|-------|-------|
| `learning-guide` | Progressive teaching through examples |
| `socratic-mentor` | Discovery learning via questioning |
| `technical-writer` | Clear, audience-tailored documentation |

---

## Analysis Depth Flags

| Flag | Tokens | When to Use |
|------|--------|-------------|
| `--think` | ~4K | 5+ files, moderate complexity |
| `--think-hard` | ~10K | Architectural analysis, system dependencies |
| `--ultrathink` | ~32K | Critical redesign, legacy modernization |

**Usage:**
```bash
/sc:analyze --think src/components/          # Standard analysis
/sc:analyze --think-hard src/                # Deep architectural analysis
/sc:troubleshoot --ultrathink "critical bug" # Maximum depth investigation
```

---

## MCP Server Flags

| Flag | Server | Best For |
|------|--------|----------|
| `--c7` / `--context7` | Context7 | Library docs, framework patterns |
| `--seq` / `--sequential` | Sequential | Multi-step reasoning, debugging |
| `--magic` | Magic (21st.dev) | UI components (`/ui`, `/21`) |
| `--morph` | Morphllm | Bulk code transformations |
| `--serena` | Serena | Session memory, large codebases |
| `--play` / `--playwright` | Playwright | Browser testing, E2E |
| `--all-mcp` | All | Maximum capability |
| `--no-mcp` | None | Native tools only |

**Usage:**
```bash
/sc:implement --c7 "add React form validation"     # With documentation lookup
/sc:troubleshoot --seq "debug auth flow"           # Multi-step reasoning
/sc:implement --magic "create dashboard component" # UI generation
/sc:cleanup --morph "rename getUserData to fetchUser across codebase"
/sc:test --playwright "test checkout flow"         # E2E browser testing
/sc:analyze --all-mcp src/                         # All servers enabled
```

---

## Execution Control

| Flag | Effect |
|------|--------|
| `--delegate [auto\|files\|folders]` | Parallel sub-agent processing |
| `--concurrency [n]` | Max concurrent operations (1-15) |
| `--loop` | Iterative improvement cycles |
| `--iterations [n]` | Set improvement cycles (1-10) |
| `--validate` | Pre-execution risk assessment |
| `--safe-mode` | Maximum validation, conservative |
| `--scope [file\|module\|project\|system]` | Analysis boundary |
| `--focus [performance\|security\|quality]` | Domain targeting |

**Usage:**
```bash
/sc:task --delegate auto "refactor entire codebase"
/sc:spawn --concurrency 10 "process all modules"
/sc:improve --loop --iterations 3 src/utils.js     # 3 improvement cycles
/sc:build --validate --safe-mode                   # Safe production build
/sc:analyze --scope project --focus security       # Project-wide security scan
```

---

## Symbol System (`--uc`)

| Symbol | Meaning |
|--------|---------|
| `->` | leads to, implies |
| `=>` | transforms to |
| `&` | and, combine |
| `\|` | or, separator |
| `>>` | sequence, then |

| Status | Meaning |
|--------|---------|
| `[x]` | completed |
| `[!]` | failed/error |
| `[~]` | in progress |
| `[?]` | warning |

| Domain | Symbol |
|--------|--------|
| Performance | `perf` |
| Security | `sec` |
| Architecture | `arch` |
| Configuration | `cfg` |

**Usage:**
```bash
/sc:analyze --uc src/                # Compressed output mode
/sc:task --uc --task-manage "large refactor"  # Token-efficient task management
```

---

## Quick Examples

```bash
# Deep analysis with documentation lookup
/sc:analyze --think-hard --c7 src/

# UI component with validation
/sc:implement --magic --validate "Add dashboard"

# Parallel task delegation
/sc:task --delegate auto "Refactor auth system"

# Safe production build
/sc:build --safe-mode --focus security

# Research with different depths
/sc:research "what is useState"                   # Quick
/sc:research "React best practices 2025"          # Standard
/sc:research --think-hard "distributed systems"   # Deep
/sc:research --ultrathink "quantum algorithms"    # Exhaustive

# Combined flags
/sc:implement --think-hard --c7 --validate "Add auth middleware"
/sc:cleanup --morph --delegate auto "standardize error handling"
```

---

## Priority Rules

1. **Safety**: `--safe-mode` > `--validate` > optimization
2. **Depth**: `--ultrathink` > `--think-hard` > `--think`
3. **MCP**: `--no-mcp` overrides individual server flags
4. **Scope**: system > project > module > file
