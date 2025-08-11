## Claude Code Subagents

Claude Code subagents are specialized tools designed to handle complex, multi-step tasks autonomously. A key benefit of Claude Code subagents is that uses its own context window separate from the main conversation and can use it's own custom prompt. Learn more about [subagents in the official documentation](https://docs.anthropic.com/en/docs/claude-code/sub-agents).

### memory-bank-synchronizer

- **Purpose**: Synchronizes memory bank documentation with actual codebase state, ensuring architectural patterns in memory files match implementation reality
- **Location**: `.claude/agents/memory-bank-synchronizer.md`
- **Key Responsibilities**:
  - Pattern documentation synchronization
  - Architecture decision updates  
  - Technical specification alignment
  - Implementation status tracking
  - Code example freshness validation
  - Cross-reference validation
- **Usage**: Proactively maintains consistency between CLAUDE-*.md files and source code to ensure documentation remains accurate and trustworthy

### code-searcher

- **Purpose**: A specialized agent for efficiently searching the codebase, finding relevant files, and summarizing code. Supports both standard detailed analysis and optional [Chain of Draft (CoD)](https://github.com/centminmod/or-cli/blob/master/examples/example-code-inspection-prompts3.md) ultra-concise mode when explicitly requested for 80% token reduction
- **Location**: `.claude/agents/code-searcher.md`
- **Key Responsibilities**:
  - Efficient codebase navigation and search
  - Function and class location
  - Code pattern identification
  - Bug source location assistance
  - Feature implementation analysis
  - Integration point discovery
  - Chain of Draft (CoD) mode for ultra-concise reasoning with minimal tokens
- **Usage**: Use when you need to locate specific functions, classes, or logic within the codebase. Request "use CoD", "chain of draft", or "draft mode" for ultra-concise responses with ~80% fewer tokens
  - **Standard mode**: "Find the payment processing code" → Full detailed analysis
  - **CoD mode**: "Find the payment processing code using CoD" → "Payment→glob:*payment*→found:payment.service.ts:45"

## Claude Code Slash Commands

### `/anthropic` Commands

- **`/apply-thinking-to`** - Expert prompt engineering specialist that applies Anthropic's extended thinking patterns to enhance prompts with advanced reasoning frameworks
  - Transforms prompts using progressive reasoning structure (open-ended → systematic)
  - Applies sequential analytical frameworks and systematic verification with test cases
  - Includes constraint optimization, bias detection, and extended thinking budget management
  - Usage: `/apply-thinking-to @/path/to/prompt-file.md`

- **`/convert-to-todowrite-tasklist-prompt`** - Converts complex, context-heavy prompts into efficient TodoWrite tasklist-based methods with parallel subagent execution
  - Achieves 60-70% speed improvements through parallel processing
  - Transforms verbose workflows into specialized task delegation
  - Prevents context overflow through strategic file selection (max 5 files per task)
  - Usage: `/convert-to-todowrite-tasklist-prompt @/path/to/original-slash-command.md`

- **`/update-memory-bank`** - Simple command to update CLAUDE.md and memory bank files
  - Usage: `/update-memory-bank`


### `/documentation` Commands

- **`/create-readme-section`** - Generate specific sections for README files with professional formatting
  - Creates well-structured sections like Installation, Usage, API Reference, Contributing, etc.
  - Follows markdown best practices with proper headings, code blocks, and formatting
  - Analyzes project context to provide relevant content
  - Matches existing README style and tone
  - Usage: `/create-readme-section "Create an installation section for my Python project"`


### `/architecture` Commands

- **`/explain-architecture-pattern`** - Identify and explain architectural patterns in the codebase
  - Analyzes project structure and identifies design patterns
  - Explains rationale behind architectural decisions
  - Provides visual representations with diagrams
  - Shows concrete implementation examples
  - Usage: `/explain-architecture-pattern`

### `/promptengineering` Commands

- **`/convert-to-test-driven-prompt`** - Transform requests into Test-Driven Development style prompts
  - Defines explicit test cases with Given/When/Then format
  - Includes success criteria and edge cases
  - Structures prompts for red-green-refactor cycle
  - Creates measurable, specific test scenarios
  - Usage: `/convert-to-test-driven-prompt "Add user authentication feature"`

- **`/batch-operations-prompt`** - Optimize prompts for multiple file operations and parallel processing
  - Identifies parallelizable tasks to maximize efficiency
  - Groups operations by conflict potential
  - Integrates with TodoWrite for task management
  - Includes validation steps between batch operations
  - Usage: `/batch-operations-prompt "Update all API calls to use new auth header"`

### `/refactor` Commands

- **`/refactor-code`** - Analysis-only refactoring specialist that creates comprehensive refactoring plans without modifying code
  - Analyzes code complexity, test coverage, and architectural patterns
  - Identifies safe extraction points and refactoring opportunities
  - Creates detailed step-by-step refactoring plans with risk assessment
  - Generates timestamped reports in `reports/refactor/` directory
  - Focuses on safety, incremental progress, and maintainability
  - Usage: `/refactor-code`