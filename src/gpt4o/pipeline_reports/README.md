# Legacy GPT-4o Pipeline Reports

`gpt4o.pipeline_reports` exists for compatibility with the pre-refactor report
structure. The real report builders now live inside the `gpt4o.pipeline`
package; this module simply forwards imports so older tooling keeps working.

## Structure

- `__init__.py` – re-exports the modern helpers (e.g.,
  `gpt4o.pipeline.reports.generate_reports`) when the package is imported.

No additional modules live here by design. Add new report functionality to the
pipeline package and expose it through this shim only when needed for
compatibility.
