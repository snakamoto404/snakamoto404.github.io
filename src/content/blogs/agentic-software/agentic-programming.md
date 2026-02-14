---
title: "Agentic Programming"
date: 2026-02-14
summary: "What I learned about programming by trying to program agents — and why the OS analogy isn't a metaphor."
---

I started using [OpenClaw](https://github.com/openclaw/openclaw) expecting a smarter chat interface. Something like the web-based assistants I'd used before — Claude, ChatGPT — but persistent, with memory, tools, and automation. I'd talk to it, it would do things. Magic unicorn.

Two days in, I realized I was wrong about what I was doing. Not wrong about what the system *could* do — wrong about the activity itself. I wasn't "chatting with an AI." I was programming. And the programming was subject to the same design principles, the same failure modes, and the same hard-won mindset that applies to any serious software.

This post is about that realization and where it leads. It connects to Andrej Karpathy's framing of [Software 3.0](https://www.youtube.com/watch?v=LCEmiRjPEtQ) — programming in natural language — but I arrived at it through practice before I had a name for it. The patterns I'll describe aren't theoretical proposals. They're things I discovered by breaking my own system and figuring out why.

### Contents

- [From chat to programming](#from-chat-to-programming)
- [The program/data separation](#the-programdata-separation)
- [The kernel emerges](#the-kernel-emerges)
- [Dangling pointers and other familiar bugs](#dangling-pointers-and-other-familiar-bugs)
- [Where we are: a kernel-only OS with raw memory access](#where-we-are-a-kernel-only-os-with-raw-memory-access)
- [What classical software tells us to build next](#what-classical-software-tells-us-to-build-next)
- [Caveats and differences](#caveats-and-differences)
- [The connection to Software 3.0](#the-connection-to-software-30)
- [What I'm betting on](#what-im-betting-on)

---

## From chat to programming

I've been using AI coding tools for a while — Claude's web interface, Claude Code, Codex. These are powerful, but they share a common model: the session is the primary unit. You open a conversation, you get output, the conversation ends (or gets compacted). Context is transient by design.

OpenClaw felt different from the start. The default `AGENTS.md` — a configuration file that ships with every workspace — <u>treats context and persistent data as first-class citizens</u>. Sessions come and go, but the workspace endures. Files persist between conversations. Instructions carry across sessions. The system has *state* in a way that a chat window doesn't.

That inversion was the first mindset shift: **sessions are not the program. Sessions are processes running on top of something more permanent.** The workspace is the operating environment. Sessions are programs that execute within it, reading from and writing to shared state.

Once I saw it that way, prompts stopped looking like "messages to an AI" and started looking like **program data** — inputs to a process that reads them, interprets them against its loaded context, and produces side effects.

This isn't just a metaphor. It changes how you work. Instead of optimizing individual conversations, you start optimizing the *environment* that all conversations share: what gets loaded, how state is organized, what invariants hold across sessions.

---

## The program/data separation

The second shift came from a frustration.

OpenClaw's default behavior dumps memory as flat, dated markdown files and maintains a long-term `MEMORY.md` at the top level. This works at small scale. But as usage increases, the mix starts to hurt. Behavioral instructions — "how to behave in this situation" — get interleaved with session artifacts — "here's what happened on Tuesday." The agent treats both the same way: it reads them into context and acts on whatever it finds.

The problem is that these are fundamentally different things. One is **program instruction** (behavioral specs, invariants, routing logic). The other is **program data** (conversation summaries, reports, artifacts). Mixing them is like storing your `.py` files in the same directory as your `.parquet` files with no distinction — and then having your runtime execute whatever it finds.

I started seeing the consequences directly. An agent would pick up a previous session's output and treat it as a current instruction. Behavioral drift would creep in as old summaries influenced new decisions in unintended ways. The system's "personality" would shift between sessions based on which memory fragments happened to load.

So I separated them. The workspace now has three distinct stores:

```
operations/   → behavioral instructions (the "program")
docs/         → persistent reference data (the "database")  
memory/       → runtime artifacts, session journals (the "filesystem")
```

This is the same distinction that operating systems enforce between text segments and data segments during program execution. It's not a cosmetic preference — it's a correctness property. Instructions should not be mutated by runtime output. Data should not be confused with executable behavior.

Behavioral consistency across sessions stabilized. Debugging became tractable — if the agent is misbehaving, I know to look in `operations/`, not in a sprawling memory dump. And new automations became easier to add because the instruction surface was clean and bounded.

---

## The kernel emerges

It was in the process of *writing* the meta-instructions — the rules about how the agent should manage its own state — that the OS analogy stopped being a loose parallel and became an operational reality.

I'd started codifying what I called **metaprogramming invariants**: rules not about what the agent should *do*, but about how it should *organize itself*. Things like:

- "Every routing README must include update hooks for self-maintenance"
- "No flat markdown files directly under `memory/` — use date folders"
- "Changes to `operations/` (behavioral store) require confirmation; changes to `memory/` (artifacts) do not"
- "Cross-file modifications must include a reference sweep for stale pointers"

These aren't task instructions. They're rules about the integrity of the instruction system itself. They're **kernel-level concerns**: memory layout enforcement, write permissions, reference integrity.

The `AGENTS.md` file — which I started interpreting as a **bootloader** — prescribes a deterministic load order on every session start:

```
1. Read SOUL.md              (firmware / behavioral identity)
2. Read USER.md + IDENTITY.md (hardware config / who's who)
3. Read operations/MEMORY.md  (kernel state / active goals)
4. Read memory/{today}        (recent runtime context)
5. Read operations/README.md  (route into active program)
```

That's an `init` sequence. Not figuratively — operationally. The order matters. Skip step 3 and the agent doesn't know what it was working on. Load step 5 before step 1 and the behavioral identity isn't established before instructions execute. Just like a real boot sequence, getting this wrong doesn't produce an error message — it produces subtly wrong behavior that's hard to diagnose.

*Our current metaprogramming invariants:*

1. **Boot sequence integrity** — deterministic load order; all referenced files must exist and be non-empty
2. **README routing invariant** — every directory under `operations/` has a README.md with routing context + update hooks
3. **Store separation** — `operations/` is behavioral (confirm before write), `docs/` is reference (inform on write), `memory/` is artifacts (write freely)
4. **No flat memory root files** — all memory artifacts live in `memory/YYYY-MM-DD/` date folders, not as root-level markdown
5. **Reference integrity** — no dangling pointers (instructions referencing files that don't exist)
6. **Delegation-by-reference** — callers point to canonical sources; don't copy behavioral text across files
7. **Update callbacks** — callers include "what to update here if the callee changes"
8. **Cross-file sweep on modification** — changes to referenced paths must include a sweep of all referencing files
9. **Ops feedback is a pointer, not a store** — operational files in `operations/` route to `memory/` for dated artifacts
10. **Cron payloads are minimal pointers** — behavioral specs live in README.md, not in cron job definitions

---

## Dangling pointers and other familiar bugs

As the metaprogramming invariants grew more complex, I started needing to break instructions into smaller, composable pieces — the equivalent of factoring functions out of a monolithic main loop. Naturally, this introduced **references**: one instruction file pointing to another for delegated behavior.

And with references came **dangling pointers**.

This wasn't abstract. Our invariant checker started surfacing concrete failures:

```
Reference Integrity (No Dangling Pointers): FAIL
- instructions reference docs/reports/README.md → file doesn't exist
- memory conventions reference operations/secretary/feedbacks.md → moved to memory/
```

Files get renamed. Sections get reorganized. An instruction that pointed to `docs/reports/README.md` keeps pointing there after the file is deleted or relocated. The agent follows the pointer, finds nothing, and either halts or — worse — confabulates a response based on the missing context.

After repairs and reference sweeps, the same check passes:

```
Reference Integrity (No Dangling Pointers): PASS
```

That fail → fix → pass cycle is exactly what static analysis feels like in a traditional codebase. And it wasn't something I designed upfront — it was something I had to build reactively, because the system kept breaking in this specific way.

Other familiar failure modes showed up naturally:

**Instructions can become garbage.** When nothing points to a behavioral spec anymore — no README routes to it, no automation references it — it sits there occupying conceptual space without being reachable. It's dead code. In principle it's harmless; in practice, it creates confusion when the agent stumbles across it during broad context loads. We need garbage collection.

**Conflicting instructions produce undefined behavior.** Two files specify contradictory rules for the same domain. The agent picks one based on load order, context window position, or what amounts to chance. This is not a prompting problem — it's a consistency problem, and the fix is the same as in any system: detect conflicts statically, before execution.

**Context overflow is a segfault.** When a task exceeds the model's context window, execution terminates abruptly:

```
context_length_exceeded: Your input exceeds the context window of this model.
```

The process didn't produce bad output. It *crashed*. Just like a segfault — the system tried to address memory it doesn't have, and the runtime killed it.

---

## Where we are: a kernel-only OS with raw memory access

If I'm honest about the current state of how I operate OpenClaw, here's the accurate analogy: **it's a kernel-only operating system with C-level raw access to memory and no enforced process isolation.**

The main agent loop handles everything — orchestration, task execution, memory management, automation triggers. When I ask it to run all tasks together, that's the kernel program doing everything. There's no scheduler, no process isolation, no protected memory.

Subagents exist and provide a form of isolation — each gets its own context (program memory) and scoped instructions (program data). But the isolation isn't enforced by the runtime. A subagent can write to shared state. Multiple subagents can trample each other's outputs. There are no locks, no capability restrictions, no access control beyond social convention ("please check after the other agent finishes").

This is powerful but dangerous. Exactly the way C gives you raw pointers: maximum flexibility, maximum footgun potential.

And the development experience reflects this. Catching stale references is time-consuming and happens on a "by-catch" basis — I notice problems when they cause visible failures, not through systematic checking. Conflicting instructions surface as confusing agent behavior that takes real debugging to trace back to the root cause. There's no compiler warning, no linter flag, no type error at write time.

---

## What classical software tells us to build next

This is where the analogy becomes not just descriptive but **predictive**. If we're really operating in a regime analogous to early systems programming — powerful but unguarded — then we can look at what classical software engineering built over the following decades and ask: which of these would help right now?

**Linting and static analysis.** We already run invariant checks for reference integrity, store separation, and routing consistency. These are hand-built lint rules. The obvious next step is making them systematic, fast, and part of every write operation — not periodic sweeps that catch problems after the fact.

**Type checking.** When one instruction delegates to another, there's an implicit contract: what the callee expects, what it returns, what it modifies. Right now these contracts are prose-level and unverified. Formalizing them — even lightly — would catch a class of integration errors that currently surface only at runtime.

**Garbage collection.** Unreachable instructions accumulate. Manual cleanup works at small scale but doesn't scale with system complexity. An automated pass that identifies instruction files with no inbound references (no README routes to them, no automation invokes them) would be a direct analogue of GC — and probably not hard to build.

**Process isolation.** Subagents should have scoped write permissions, bounded resource budgets, and explicit IPC channels. Not "please don't write to that file" — actual enforcement. This is the containers/namespaces move for agent execution.

**CI/CD for behavioral specs.** Version-controlled instructions should be testable against regression scenarios before deployment. "I changed how the morning brief automation works" should trigger a validation suite, not a prayer.

These aren't speculative futurism. They're concrete tools that classical software engineering proved necessary under exactly the same pressures: growing system complexity, unreliable execution substrates, and the need for multiple contributors to work on shared state without breaking each other.

---

## Caveats and differences

The analogy isn't perfect, and it's worth being explicit about where it strains.

**Stochastic execution.** Classical CPUs are deterministic at the instruction level. LLMs are not. The same prompt can produce different outputs. This means invariants need to be checked *after* execution, not just before — you can't guarantee behavior by guaranteeing input. This strengthens the case for runtime checks and verification, but it also means some classical techniques (like deterministic replay) don't transfer directly.

**Natural language ambiguity.** Code has formal semantics. Prompts don't. Two reasonable people can read the same instruction and interpret it differently — and so can two model runs. This makes "type checking" harder: you're checking contracts expressed in prose, not in a formal type system. It's more like linting natural-language specifications than like running `mypy`.

**The substrate is improving.** Models get better. Context windows grow. Reasoning improves. Some of today's failure modes may recede. But I'd argue this makes the architectural work *more* important, not less — the same way faster CPUs didn't eliminate the need for operating systems. They made it possible to run bigger, more complex systems, which made architecture *more* critical.

**This is one operator's experience.** I'm describing patterns from a single workspace with a specific set of automations. The failure modes are real, but I don't have population-level data. Treat this as a heavily instrumented case study, not a controlled experiment.

---

## The connection to Software 3.0

Karpathy's ["Software Is Changing (Again)"](https://www.youtube.com/watch?v=LCEmiRjPEtQ) keynote articulates the same shift from a different angle. His taxonomy — Software 1.0 (code), 2.0 (trained weights), 3.0 (natural-language prompts) — names what I was experiencing without having a label for it ([01:39–03:25](https://www.youtube.com/watch?v=LCEmiRjPEtQ&t=99s)).

His LLM-as-operating-system framing — model as CPU, context window as RAM, tools as peripherals ([09:12–10:57](https://www.youtube.com/watch?v=LCEmiRjPEtQ&t=552s)) — maps directly to what I was building, though I was approaching it from the operator side rather than the product-design side.

Two of his points resonate especially strongly with what I described above:

1. **"We are in the 1960s"** ([11:04](https://www.youtube.com/watch?v=LCEmiRjPEtQ&t=664s)): centralized, expensive compute with immature abstractions. If that's right, then the playbook for what comes next already exists — schedulers, filesystems, type systems, debuggers, package managers. The substrate changed. The engineering logic didn't.

2. **"Not the year of agents — the decade of agents"** ([27:34](https://www.youtube.com/watch?v=LCEmiRjPEtQ&t=1654s)): autonomy is a slider, not a switch. The systems architecture — memory management, verification interfaces, isolation primitives — doesn't become obsolete as models improve. It becomes the infrastructure that lets you *safely increase* the autonomy setting.

I use Karpathy here as triangulation, not authority. I arrived at these patterns through debugging my own workspace. His framing independently converges on the same terrain. That convergence is what gives me confidence the patterns are real and not just one person's idiosyncratic setup.

---

## What I'm betting on

The practical takeaway is simple: if you're building agentic systems, study operating systems and programming language design as seriously as you study prompting.

The failure modes are catalogued. The engineering playbooks exist. The abstractions are waiting to be translated into this new medium — carefully, concretely, and with the epistemic humility that comes from knowing your execution substrate is partly made of stochastic language.

We're in the early days of a long buildout. The question isn't whether these tools will be needed. It's who builds them first.
