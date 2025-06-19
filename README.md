# MEA-analysis

Study of the influence of neuronal sialylation on network activity using human brain tissue and a multielectrode array.

---

`fileprocmax2.py` prepares data into more manageable formats.

**Workflow:**
`.mcd` --(MC_DataTool)--> `(ASCII) .txt` --(`fileprocmax2.py`)--> `.tsv` (This output then goes to `k45.ipynb` or `g20.py`)

---

`k45.ipynb` is the main code that performs most of the analysis.
* Modify `config.json` to change main variables. (It will create one if you're missing it.)

---

`g20.py` creates network activity comparison graphs for many channels.

---

**Note:** Supplied data has only 3500ms because it is very dense (@20kHz).
