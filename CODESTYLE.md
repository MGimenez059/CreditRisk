# CreditRisk — Code Style & Engineering Standards

This document defines the engineering standards for the CreditRisk codebase. It is binding for all code, documentation, comments, commits, and artifacts produced in this repository. The goal is a codebase that behaves like production-ready, enterprise-grade software: predictable, testable, secure, and safe to change.

If a rule in this document conflicts with a tool's default behavior, the tool configuration must be updated to match this document — not the other way around.

---

## 1. Language Policy

**English is the only language used anywhere in this repository.** No exceptions.

This applies to:

- Code: variable, function, class, module, and file names
- Comments and docstrings
- Commit messages, branch names, PR titles and descriptions
- Log messages and error messages (including user-facing API error responses)
- Documentation (`README.md`, `docs/`, `ROADMAP.md`, ADRs, model card, data dictionary)
- Test names and test descriptions
- Configuration keys, environment variable names, and CLI help text

Rationale: mixed-language codebases fragment searchability, break tooling (linters, spellcheckers, IDE navigation), and create ambiguity for automated tooling and future contributors. There is no scenario in this project where a non-English artifact is acceptable, including scratch scripts, notebooks, or internal-only tooling.

---

## 2. Guiding Principles

1. **Clarity over cleverness.** Code is read far more often than it is written. Prefer the obvious solution over the clever one.
2. **Explicit over implicit.** No magic. No hidden side effects. No implicit type coercion. State intent directly.
3. **Fail fast, fail loud.** Invalid state should raise immediately at the boundary where it's detected, not propagate silently.
4. **Single Responsibility.** Every module, class, and function should have one reason to change.
5. **Composition over inheritance.** Prefer small composable units and dependency injection over deep class hierarchies.
6. **YAGNI, but design for extension at real seams.** Don't build speculative abstractions; do keep layer boundaries (API / service / repository / ML) clean so extension is cheap when it's actually needed.
7. **Deterministic and reproducible by default.** Given the same input, config, and seed, the system must produce the same output — this applies to data pipelines, training, and inference alike.
8. **Boring is a feature.** Favor well-understood patterns (repository, service layer, dependency injection) over novel or clever architectures.

---

## 3. Architecture & Layering

The project follows a strict layered (clean) architecture. Dependencies point **inward**; outer layers know about inner layers, never the reverse.

```text
api/            → HTTP concerns only (routing, request/response, status codes)
schemas/        → Pydantic I/O contracts (API boundary DTOs)
services/       → Business logic, orchestration
repositories/   → Data access abstraction (SQLAlchemy queries live here only)
db/             → ORM models, session, engine
ml/             → Training, inference, preprocessing, explainability (framework-agnostic)
```

**Hard rules:**

- Route handlers (`api/routes/*.py`) contain **no business logic**. They validate input via Pydantic, call a service, and map the result to a response schema.
- SQLAlchemy models never leave the `repositories`/`db` layer. API responses are built from Pydantic schemas, never from ORM objects directly.
- Services depend on repository **interfaces**, not concrete SQLAlchemy queries, so they can be tested with fakes/mocks.
- `ml/` code must be importable and testable with no FastAPI or database dependency — training and inference logic must run standalone (e.g., from `scripts/train_model.py`) with no web framework in the import path.
- Cross-layer shortcuts (e.g., a route querying the DB directly) are not acceptable, even for "quick" endpoints.

```python
# ❌ Bad — business logic and ORM access inside the route handler
@router.post("/predictions")
async def create_prediction(payload: PredictionRequest, db: Session = Depends(get_db)):
    features = [payload.age, payload.income]  # feature logic leaking into API layer
    model = joblib.load("models/model.joblib")  # I/O and ML concerns in the route
    prob = model.predict_proba([features])[0][1]
    db.execute("INSERT INTO predictions ...")  # raw SQL in the route
    return {"probability": prob}
```

```python
# ✅ Good — route delegates to a service; layers stay separated
@router.post("/predictions", response_model=PredictionResponse)
async def create_prediction(
    payload: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    result = await prediction_service.predict(payload)
    return PredictionResponse.from_domain(result)
```

---

## 4. Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Modules / files | `snake_case` | `prediction_service.py` |
| Classes | `PascalCase` | `PredictionService`, `CustomerRepository` |
| Functions / methods | `snake_case`, verb-first | `calculate_risk_score()`, `get_active_model()` |
| Variables | `snake_case`, descriptive noun | `default_probability`, `credit_utilization` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_RANDOM_STATE`, `MAX_BATCH_SIZE` |
| Booleans | `is_` / `has_` / `should_` prefix | `is_valid`, `has_previous_default` |
| Pydantic schemas | Suffix by role | `PredictionRequest`, `PredictionResponse`, `CustomerCreate` |
| SQLAlchemy models | Singular noun | `Customer`, `Loan`, `Prediction` |
| Exceptions | Suffix `Error` | `ModelNotFoundError`, `InvalidFeatureSchemaError` |
| Test files | `test_<module>.py` | `test_risk_service.py` |
| Test functions | `test_<unit>_<scenario>_<expected>` | `test_risk_score_below_threshold_returns_low()` |

**Forbidden:**

- Generic names: `data`, `temp`, `obj`, `manager`, `helper`, `stuff`, `handler2`
- Abbreviations that aren't domain-standard (`cust` for customer, `pred` for prediction) — spell it out
- Hungarian notation or type suffixes (`str_name`, `intAge`)
- Single-letter identifiers outside of trivial, tightly-scoped loop indices or well-known math notation (e.g., `X`, `y` in ML training code is acceptable as it matches scikit-learn convention)

---

## 5. Python Style & Formatting

- **Formatter/linter:** Ruff (format + lint), configured in `pyproject.toml`. No manual formatting debates — `ruff format` is authoritative.
- **Line length:** 100 characters.
- **Import order:** stdlib → third-party → first-party (`credit_risk.*`), each group alphabetized, enforced by Ruff's isort ruleset. No wildcard imports (`from x import *`).
- **Type hints are mandatory** on every function signature (parameters and return type), including private/internal functions. `Any` requires a comment justifying why a concrete type isn't possible.
- Use `from __future__ import annotations` is unnecessary on Python 3.12+; use native generic syntax (`list[str]`, `dict[str, int]`, `X | None`).
- Prefer `pathlib.Path` over `os.path`.
- Prefer f-strings over `.format()` or `%`.
- No mutable default arguments (`def f(x: list = [])` is forbidden).
- Dataclasses (`@dataclass(frozen=True)` where possible) for internal value objects; Pydantic models only at I/O boundaries (API schemas, config, external data contracts).

```python
# ❌ Bad
def calc(a, b=[]):
    b.append(a)
    return b

# ✅ Good
def append_to_history(value: float, history: list[float] | None = None) -> list[float]:
    history = history if history is not None else []
    history.append(value)
    return history
```

---

## 6. Type Safety

- MyPy runs in **strict mode** (`strict = true` in `pyproject.toml`). CI fails on any type error.
- No `# type: ignore` without a trailing comment explaining why (`# type: ignore[arg-type]  # XGBoost stubs don't cover DMatrix`).
- Use `Protocol` to define repository/service interfaces consumed by other layers, so implementations can be swapped or mocked without inheritance coupling.
- Use `Literal` for closed sets of string values (e.g., `risk_level: Literal["LOW", "MEDIUM", "HIGH"]`) instead of bare `str`.
- Avoid `Optional[X]` sprawl: if a field is "always present after validation," model that with non-optional types and validate at the boundary instead of pushing `None`-checks downstream.
- Every public function must be understandable from its signature alone, without reading the body.

---

## 7. Docstrings

All public modules, classes, and functions require a docstring in **Google style**. Private helper functions (prefixed `_`) require a docstring only if their behavior is non-obvious from the name and signature.

```python
def calculate_risk_score(default_probability: float) -> RiskScore:
    """Convert a model-predicted default probability into a risk score.

    Args:
        default_probability: Predicted probability of default, in [0, 1].

    Returns:
        A RiskScore containing the numeric score (0-100) and the
        corresponding risk level (LOW, MEDIUM, HIGH).

    Raises:
        ValueError: If default_probability is outside [0, 1].
    """
```

Docstrings describe **contract**, not implementation: what the function guarantees, its inputs/outputs, and the conditions under which it raises. They do not restate the function body line by line.

---

## 8. Comments Policy

Comments explain **why**, never **what**. If a comment restates what the next line already says, delete the comment.

```python
# ❌ Bad — restates the obvious
# increment counter by 1
counter += 1

# ❌ Bad — vague, adds no information
# fix for bug
if value < 0:
    value = 0

# ✅ Good — explains non-obvious reasoning
# XGBoost's scale_pos_weight expects the negative/positive class ratio,
# not the raw class counts — see docs/model_card.md#class-imbalance
scale_pos_weight = negative_count / positive_count

# ✅ Good — documents a constraint that isn't visible locally
# Must run before `_validate_schema`: dtype coercion changes null
# representation from empty-string to NaN, which the validator expects.
df = _coerce_dtypes(df)
```

**Rules:**

- No commented-out code. Delete it — Git history preserves it.
- No `TODO`/`FIXME` without an issue reference (`# TODO(CRISK-142): support CSV batch upload`).
- No comments that apologize for or narrate bad code ("hacky but works", "don't touch this"). If code needs that disclaimer, fix the code instead.
- Comments on business rules should reference the spec/domain source when the rule isn't self-evident from the code (e.g., why a threshold is 0.42 and not 0.5).

---

## 9. Error Handling

- Define a project-specific exception hierarchy rooted at a single base (`CreditRiskError`), with subclasses per domain concern (`ModelNotFoundError`, `InvalidFeatureSchemaError`, `DataValidationError`, `EntityNotFoundError`).
- **Never use bare `except:`.** Catch the narrowest exception type that can actually occur.
- **Never silently swallow exceptions.** A caught exception is either handled meaningfully, re-raised, or re-raised as a more specific domain exception (`raise ModelLoadError(...) from err`) — never `pass`.
- API layer translates domain exceptions into HTTP responses via a centralized exception handler (FastAPI `exception_handler`), not per-route `try/except` blocks.
- Validate inputs at the boundary (Pydantic) and fail with `HTTP 422` before any business logic runs.
- Data pipeline validation failures **stop the pipeline** — they must never be logged and ignored, or silently coerced.

```python
# ❌ Bad
try:
    model = load_model(path)
except Exception:
    model = None  # silently continues with no model

# ✅ Good
try:
    model = load_model(path)
except FileNotFoundError as err:
    raise ModelNotFoundError(f"No model artifact found at '{path}'") from err
```

---

## 10. Logging

- Use structured logging (`structlog` or stdlib `logging` with a JSON formatter) — never `print()` in application code.
- Every log line in a request path includes a **request/correlation ID** to allow tracing a single prediction end-to-end.
- Log levels are used with intent:
  - `DEBUG` — internal state useful only during development
  - `INFO` — lifecycle events (startup, model loaded, prediction served)
  - `WARNING` — recoverable anomalies (e.g., fallback threshold used)
  - `ERROR` — failed operations requiring attention
- **Never log**: raw customer input payloads in full, secrets, database credentials, or anything resembling PII — even though the dataset is synthetic/public, the codebase must behave as if it were handling real data.
- Log the **model version** and **prediction latency** on every prediction, per spec §33.

```python
logger.info(
    "prediction_served",
    request_id=request_id,
    model_version=model.version,
    latency_ms=latency_ms,
    risk_level=result.risk_level,
)
```

---

## 11. Testing Standards

- Test framework: **Pytest**. Structure: `tests/unit/`, `tests/integration/`, `tests/fixtures/`.
- Follow **Arrange-Act-Assert** structure in every test; one logical assertion focus per test.
- Test names describe unit, scenario, and expected outcome: `test_risk_score_probability_above_71_returns_high()`.
- Unit tests have **no I/O**: no real database, no filesystem, no network. Mock repositories and external boundaries.
- Integration tests use a dedicated test database/container (never the dev or production database) and are isolated per test (transaction rollback or fresh schema per run).
- Tests must be **independent and order-agnostic** — no test may depend on state left by another.
- ML-specific tests are mandatory (per spec §27): pipeline trains end-to-end on a small fixture dataset, output probabilities are within `[0, 1]`, the inference feature schema matches the training schema, and no unexpected `NaN` reaches the model.
- No `assert True` placeholder tests. No skipped tests without a linked issue explaining why.
- Coverage is a signal, not a target to game: prioritize covering business logic, edge cases, and failure paths over chasing a percentage.

---

## 12. API Design Conventions

- Base path is versioned: `/api/v1`. Breaking changes require a new version, not in-place mutation of a contract.
- Resource-oriented, plural nouns: `/predictions`, `/customers`, `/models`. Actions that don't map to CRUD use a sub-resource verb (`/predictions/batch`), not a query parameter flag.
- Request/response bodies are always Pydantic schemas — never raw dicts, never ORM models serialized directly.
- Standard status codes: `200` success, `201` created, `422` validation error, `404` not found, `409` conflict, `500` unhandled server error (should be rare and always logged with a stack trace).
- Every endpoint that returns model-derived output includes the **model name and version** used to produce it, so responses are always traceable to a specific artifact.

---

## 13. Database & ORM Conventions

- Tables: plural snake_case (`customers`, `credit_histories`). Columns: singular snake_case.
- Every table has `id` (UUID, PK), and `created_at` (and `updated_at` where the row is mutable).
- Foreign keys are indexed. Constraints (`NOT NULL`, `CHECK`, uniqueness) are enforced at the database level, not only in application code.
- All schema changes go through **Alembic migrations** — no manual DDL, no "sync your local DB by hand" workflows.
- Raw SQL strings are not allowed outside the `repositories/` layer; use SQLAlchemy Core/ORM constructs everywhere else.
- Transactions are managed explicitly at the service layer boundary — a service method that performs multiple writes either fully commits or fully rolls back.

---

## 14. ML Code Conventions

- `random_state=42` (or an explicitly configured seed) is set everywhere randomness is involved: splits, model init, CV folds, Optuna sampler. No unseeded randomness in training code.
- Preprocessing and feature engineering are implemented as reusable, scikit-learn-compatible transformers in `ml/`, never as one-off notebook cells duplicated into scripts.
- The full inference pipeline (preprocessing + feature engineering + model) is serialized as a single artifact so training-serving skew is structurally impossible.
- No hardcoded file paths inside `ml/` modules — paths are injected via config/settings.
- Every trained model artifact is paired with metadata (algorithm, dataset version, feature version, metrics, training timestamp) persisted alongside it, per spec §24–25.
- Training scripts (`scripts/train_model.py`) must be runnable end-to-end on a clean checkout with no manual steps beyond `pip install` and dataset download, per the reproducibility requirement in spec §30.

---

## 15. Security Practices

- No secrets, credentials, tokens, or connection strings are ever committed. `.env` is git-ignored; `.env.example` documents required variables with placeholder values only.
- All external input (API payloads, uploaded batch files) is treated as untrusted and validated before use.
- Dependencies are pinned (`pyproject.toml` lockfile) and updated deliberately, not floating on `*`.
- Database access uses parameterized queries exclusively (guaranteed by using the ORM/Core query builder, never string-interpolated SQL).
- No real PII enters the codebase, fixtures, logs, or example data — synthetic/public data only, per spec §36.

---

## 16. Configuration & Environment

- All configuration is loaded through a single `Settings` object (`pydantic-settings`) reading from environment variables — no scattered `os.environ.get()` calls throughout the codebase.
- Defaults are safe for local development only; production values are always supplied via environment, never hardcoded.
- `.env.example` is kept in sync with every configuration variable the application actually reads.

---

## 17. Git & Commit Conventions

- **Conventional Commits**, in English, imperative mood: `feat: add batch prediction endpoint`, `fix: correct risk score rounding`, `test: add coverage for threshold edge cases`.
- Allowed types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`.
- One logical change per commit. No `update`, `fix stuff`, `wip`, `final`, `final2`.
- Branch naming: `feature/<short-description>`, `fix/<short-description>`, `chore/<short-description>`.
- PRs describe **what** changed and **why**, link the relevant roadmap phase/issue, and call out any follow-up work explicitly rather than leaving it implicit.

---

## 18. Tooling & Enforcement

Enforced automatically via `pre-commit` and CI (GitHub Actions) — no rule in this document is enforced by convention alone if a tool can enforce it instead:

```text
ruff format --check   # formatting
ruff check            # linting
mypy src              # strict type checking
pytest                # unit + integration tests
```

CI fails the build on any violation. No merging with a red pipeline, and no disabling a check to "merge now, fix later."

---

## 19. Complexity & Size Guidelines

- Functions should fit on one screen (~40 lines) and do one thing. If a function needs a comment to separate "sections," it should be split into named functions instead.
- Prefer early returns / guard clauses over deeply nested conditionals.
- A module exceeding ~300 lines is a signal to reconsider its responsibilities, not a hard blocker on its own.
- Cyclomatic complexity above ~10 in a single function is a refactor signal (flagged by Ruff's complexity ruleset).

---

## 20. Pre-Merge Checklist

- [ ] All code, names, comments, docstrings, and commit messages are in English
- [ ] Types are complete; `mypy --strict` passes
- [ ] `ruff format` and `ruff check` pass with no suppressions added without justification
- [ ] No business logic in route handlers; no raw SQL outside repositories
- [ ] New/changed behavior has unit tests; boundary-crossing behavior has integration tests
- [ ] Errors are handled with specific exceptions, never swallowed
- [ ] No `print()`; logging uses structured fields and a request ID where applicable
- [ ] No secrets, PII, or hardcoded paths introduced
- [ ] Public functions/classes have Google-style docstrings; comments explain "why," not "what"
- [ ] Commit messages follow Conventional Commits
