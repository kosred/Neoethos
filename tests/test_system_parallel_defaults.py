from forex_bot.core.system import derive_live_symbol_concurrency, derive_parallel_budget_hints


def test_parallel_budget_hints_cpu_only_host() -> None:
    hints = derive_parallel_budget_hints(cpu_budget=11, gpu_count=0)

    assert hints.cpu_budget == 11
    assert hints.gpu_workers == 0
    assert hints.cpu_threads_per_gpu == 11
    assert hints.parallel_models_mode == "auto"


def test_parallel_budget_hints_gpu_host_scales_without_fixed_cap() -> None:
    hints = derive_parallel_budget_hints(cpu_budget=249, gpu_count=8)

    assert hints.cpu_budget == 249
    assert hints.gpu_workers == 8
    assert hints.cpu_threads_per_gpu == 31
    assert hints.parallel_models_mode == "auto"


def test_live_symbol_concurrency_scales_with_ram_and_cpu_budget() -> None:
    workers = derive_live_symbol_concurrency(
        symbol_count=12,
        cpu_budget=11,
        available_ram_gb=18.5,
        per_symbol_gb=4.0,
    )

    assert workers == 4


def test_live_symbol_concurrency_respects_symbol_count_floor() -> None:
    workers = derive_live_symbol_concurrency(
        symbol_count=1,
        cpu_budget=249,
        available_ram_gb=512.0,
        per_symbol_gb=4.0,
    )

    assert workers == 1
