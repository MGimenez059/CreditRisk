"""HTTP layer: FastAPI routers, request/response wiring, and dependency injection.

Route handlers in this package contain no business logic — they validate
input via Pydantic schemas, delegate to a service, and map the result back
to a response schema. See CODESTYLE.md §3.
"""
