# `qec-utility-estimation` 0.1.0

The code behind the resource estimate in the neutrino school's lecture on
quantum computing: one circuit for carbon on a small lattice, priced piece by
piece. It holds the library modules the calculation uses, their tests, and a
script that prints the calculation in the order of the slides. The slides are
in `slides/`.

Exported 2026-11-13 from commit `25a122cfbfdfe507a6044cdcc83df09a74b31871`.

## Quickstart

You need uv (https://docs.astral.sh/uv/). It fetches the Python version
pinned in `.python-version`.

    uv sync --locked                             # the environment, from the lock
    uv run python scripts/price_one_circuit.py   # the lecture's calculation
    uv run jupyter lab                           # the notebooks, in your browser
    uv run pytest                                # the tests
    uv run ruff check                            # lint

## Modules

The library lives under `src/qec_utility_estimation/`:

- `src/qec_utility_estimation/apps/pionless_first_order/couplings.py`: Pionless-EFT reference couplings at fixed lattice spacings (Watson+ Tab. 3).
- `src/qec_utility_estimation/apps/pionless_first_order/norms.py`: Pionless-EFT Hamiltonian norm inputs for qubitization.
- `src/qec_utility_estimation/apps/pionless_first_order/response_circuit.py`: The cost of one circuit of the neutrino-response calculation, piece by piece.
- `src/qec_utility_estimation/qubitization.py`: Roggero+ 2019 App. C controlled-qubiterate gate-cost model.
- `src/qec_utility_estimation/state_preparation.py`: Cost of loading a classically known sparse state onto the system register.
- `src/qec_utility_estimation/synthesis.py`: Rotation synthesis: the precision per rotation, and its T cost.

## Scripts

Each script runs with `uv run python <script>`. Try `--help` first: it says what
the script prints and which options each of the lecture's exercises uses.

- `scripts/price_one_circuit.py`: Price one circuit of the neutrino-response calculation, in the order of hour 2.

## Notebooks

Each notebook works the calculation in short cells that call the library, with
its outputs saved, so it reads on the repository's page with nothing installed.
To run one yourself, `uv run jupyter lab` opens Jupyter in your browser; open
the notebook there and run its cells from the top.

- `notebooks/price_one_circuit.ipynb`: Pricing one circuit: carbon on the 4^3 lattice

## Tests

`tests/` holds 6 test modules; the suites mirror the modules.

## Sources

Each module's docstring names its sources. The calculation's are at the top of
the module `response_circuit`, listed above, with the equation each piece rests
on. The papers:

- The Hamiltonian, its couplings and lambda: Roggero et al. 2019,
  arXiv:1911.06368.
- The quantum walk and its phase estimation: Babbush et al. 2018,
  arXiv:1805.03662, with Gidney's AND gate, arXiv:1709.06648.
- The current and its success probability: Roggero, Gu, Baroni and
  Papenbrock 2020, arXiv:2009.13485.
- Amplitude amplification: Brassard, Høyer, Mosca and Tapp 2002,
  arXiv:quant-ph/0005055.
- Loading the ground state: Fomichev et al. 2023, arXiv:2310.18410.
- Rotation synthesis: Bocharov, Roetteler and Svore 2015, arXiv:1404.5320;
  Campbell 2017, arXiv:1612.02689; Ross and Selinger 2016, arXiv:1403.2975.
- The surface code's logical error rate: Fowler et al. 2012,
  arXiv:1208.0928.
- The budgets: the U.S. Department of Energy's request for information on
  scientifically relevant fault-tolerant quantum computing systems,
  892431-26-RFI-0001 (May 2026).
