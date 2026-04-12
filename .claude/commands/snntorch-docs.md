# SNNTorch Docs

Search and read the local snntorch documentation (RST files in `docs/snntorch/docs/`).

## Arguments

`$ARGUMENTS` — a topic, class name, or keyword to look up (e.g. `Leaky`, `SLSTM`, `surrogate`, `spikegen`)

## Steps

1. If `$ARGUMENTS` is empty, list all RST files in `docs/snntorch/docs/` so the user can pick a topic.

2. Otherwise, search for `$ARGUMENTS` across the RST files:
   ```bash
   grep -ril "$ARGUMENTS" docs/snntorch/docs/
   ```

3. Read the most relevant file(s) — prioritise files whose **name** matches (e.g. `snn.neurons_leaky.rst` for `Leaky`), then files that mention the term most often.

4. Present the content in a clean, readable way:
   - Strip RST directive boilerplate (`:param`, `.. autoclass::` headers, etc.) where unhelpful
   - Keep docstrings, parameter descriptions, equations, and examples
   - Quote the source file path so the user can open it directly

5. If nothing is found, say so and suggest related terms from the file listing.

## Doc File Index (quick reference)

| File | Contents |
|------|----------|
| `snntorch.rst` | Top-level module overview |
| `snn.neurons_leaky.rst` | `Leaky` LIF neuron |
| `snn.neurons_lapicque.rst` | `Lapicque` RC neuron |
| `snn.neurons_synaptic.rst` | `Synaptic` conductance-based neuron |
| `snn.neurons_alpha.rst` | `Alpha` neuron |
| `snn.neurons_slstm.rst` | `SLSTM` spiking LSTM cell |
| `snn.neurons_sconvlstm.rst` | `SConvLSTM` spiking convolutional LSTM |
| `snn.neurons_rleaky.rst` | `RLeaky` recurrent LIF neuron |
| `snn.neurons_rsynaptic.rst` | `RSynaptic` recurrent synaptic neuron |
| `snn.neurons_leakyparallel.rst` | `LeakyParallel` (parallelisable LIF) |
| `snntorch.functional.rst` | Loss functions, regularisation, spike ops |
| `snntorch.backprop.rst` | BPTT helpers |
| `snntorch.spikegen.rst` | Spike generation utilities |
| `snntorch.spikeplot.rst` | Visualisation |
| `snntorch.export_nir.rst` | NIR export |
| `snntorch.import_nir.rst` | NIR import |
| `quickstart.rst` | Quickstart tutorial |
| `installation.rst` | Installation instructions |
| `examples.rst` | Tutorial examples index |
| `contributing.rst` | Contributing guide |
| `history.rst` | Changelog |
