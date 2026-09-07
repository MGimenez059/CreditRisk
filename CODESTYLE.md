# CreditRisk — Engineering Conventions

Practical conventions for an educational portfolio project. Keep quality controls useful and scope proportionate.

## 1. Language Policy

Use English for repository code, documentation, tests and commit messages. User conversation may use the user's language.

## 2. Guiding Principles

Prefer clarity, explicit boundaries, reproducibility and small changes. Avoid speculative abstractions and claims of enterprise readiness.

## 3. Architecture & Layering

Routes handle HTTP; services orchestrate; repositories write database queries.
Services may use ORM entities internally, but responses use Pydantic schemas.
ML modules have no FastAPI/database dependency. Use repository protocols where
fakes help testing; ordinary functions do not need interfaces. Existing dormant
customer routes must gain a service and integration coverage before activation.

## 4. Naming Conventions

Use descriptive snake_case functions/variables, PascalCase classes and UPPER_SNAKE_CASE constants. Familiar domain and numerical conventions are acceptable when clear.

## 5. Python Style & Formatting

Ruff owns formatting (100 columns), imports and lint rules. Type application and
script function signatures. Prefer pathlib and native generic syntax. Avoid mutable
defaults. Postponed annotations or quoted forward references remain valid when needed;
Python 3.12 does not make every forward reference automatically safe.

## 6. Type Safety

MyPy strict checks src and scripts. Explain necessary suppressions and avoid
unnecessary Any. Use Literal for closed values. Keep type abstractions proportional
to the boundary they protect; third-party untyped libraries need explicit accommodations.

## 7. Docstrings

Public APIs need concise contract docstrings. Google-style Args/Returns/Raises sections help nontrivial interfaces; do not repeat obvious signatures mechanically.

## 8. Comments Policy

Explain why. Avoid commented-out code and repetitive references to rules in every docstring. TODOs reference an issue or a concrete ROADMAP phase, e.g. TODO(ROADMAP-P4).

## 9. Error Handling

Raise specific exceptions and preserve causes. Structural data errors stop ingestion;
documented row exclusions are allowed with reasons and a 5% maximum. Never silently
ignore invalid state. Map domain failures to HTTP centrally.

## 10. Logging

Use structured application logs and request correlation where applicable. No raw
applicant payloads or credentials. CLI tools may print concise stdout/stderr messages.
Only claim complete request tracing once all relevant paths implement it.

## 11. Testing Standards

Use Pytest with independent fixtures. Isolated filesystem tests may use tmp_path;
unit tests do not need live databases/network. Database integration tests use a
dedicated database. Verify meaningful behavior: grouping, raw-input round trips,
preprocessing isolation, service boundaries and failure paths. Coverage is not a quota.

## 12. API Design Conventions

Version paths under /api/v1. Requests/responses use schemas and identify the actual
model artifact. Unreleased scaffolds may be corrected in place; once published,
breaking contract changes need a migration/version plan. Use synchronous routes
for the synchronous service stack.

## 13. Database & ORM Conventions

Use explicit keys, constraints, indexes and Alembic revisions. Services own
transaction orchestration; repositories own queries. Optional customer/loan scaffolds
do not need expansion for MVP. Before enabling persistence, verify atomic writes.

## 14. ML Code Conventions

Keep feature engineering + preprocessing + estimator in one serializable pipeline.
Only training rows fit learned transformations. Identical raw-input groups never
cross splits/folds. Record seeds, parameters, input hash, split positions and runtime.
Test the API adapter against the saved pipeline; serialization alone cannot eliminate
every possible training/serving mismatch.

## 15. Security Practices

Never commit real secrets or direct personal identifiers. Local demo credentials in
Compose/.env.example are not production credentials. Validate external inputs and
use parameterized database access. Commit uv.lock and update dependencies deliberately.

## 16. Configuration & Environment

Application configuration comes from Settings. CLIs may accept explicit path/run
arguments. Keep .env.example synchronized. The loaded artifact determines model
identity; defaults do not establish which model was actually selected.

## 17. Git & Commit Conventions

Use English Conventional Commits for logical changes and concise PR descriptions explaining behavior and validation. Keep unrelated changes out of the same commit.

## 18. Tooling & Enforcement

Run ruff format --check ., ruff check ., mypy src scripts and pytest in the locked
environment. CI must fail on actual errors. Do not disable checks to hide defects;
tool rules should implement agreed conventions, not create unnecessary process.

## 19. Complexity & Size Guidelines

Prefer short cohesive functions and guard clauses. Roughly 40 lines/function, 300/module and complexity 10 are review signals, not reasons to invent extra layers.

## 20. Pre-Merge Checklist

Before completion: relevant tests pass, formatting/lint/types pass, documentation
matches actual behavior, no secrets are introduced, and remaining execution limitations
are explicit. A trained result requires a real run, not a plausible table.
