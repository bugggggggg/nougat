

Also change the code in **python3.9/site-packages/pypdfium2/_helpers/document.py** to forbid multirocessing rendering.

```python
pool_kwargs = dict(
    initializer = _parallel_renderer_init,
    initargs = (self._input, self._password, bool(self.formenv), mk_formconfig, renderer, converter, converter_params, pass_info, kwargs),
)
with mp.Pool(n_processes, **pool_kwargs) as pool:
    yield from pool.imap(_parallel_renderer_job, page_indices)
```

to 
```python
if n_processes > 1:
    pool_kwargs = dict(
        initializer = _parallel_renderer_init,
        initargs = (self._input, self._password, bool(self.formenv), mk_formconfig, renderer, converter, converter_params, pass_info, kwargs),
    )
    with mp.Pool(n_processes, **pool_kwargs) as pool:
        yield from pool.imap(_parallel_renderer_job, page_indices)
else:
    _parallel_renderer_init(self._input, self._password, bool(self.formenv), mk_formconfig, renderer, converter, converter_params, pass_info, kwargs)
    for index in page_indices:
        yield _parallel_renderer_job(index)
```