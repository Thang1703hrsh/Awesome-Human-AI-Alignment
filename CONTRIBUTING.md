# Contributing

Contributions may extend the research catalog, software adapters, or assurance procedures. Keep these layers distinct so that a paper entry does not imply an executable implementation and an implementation does not imply independent evidence of alignment.

## Research catalog changes

- Use an existing terminal category unless the survey taxonomy itself has been revised.
- Multi-label papers only when each placement reflects a distinct functional role.
- Preserve stable citation keys and include a primary publication or preprint URL.
- Run `hai-align catalog validate` before opening a pull request.

## Software changes

- Implement one of the capability-specific protocols documented in `docs/CODEBASE.md`.
- Do not import optional ML frameworks from the package root.
- Validate inputs and return the standard result dataclass.
- Record software coverage accurately in `catalog/data/methods.json`: use `adapter`
  for user-supplied implementations, `reference` for small bundled algorithms, and
  `native` for first-class wrappers with validation, documentation, and tests.
- Add focused unit tests and avoid network calls in the default test suite.

## Local checks

```bash
python -m pip install -e .
python -m compileall -q src tests examples
python -m unittest discover -s tests -v
hai-align catalog validate
```
