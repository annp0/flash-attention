The project was setup with [uv](https://docs.astral.sh/uv/).

To compile the kernel and run the benchmark, execute:
```
uv run python3 main.py
```

If you encounter issues, run the following command and include the output in your bug report:

```
compute-sanitizer -- uv run python3 main.py
```