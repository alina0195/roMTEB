# Reluare eval / tabele / figuri RoMTEB

Toate comenzile din rădăcina repo-ului
(`/export/projects/nlp/users/alina_gheorghe/roMTEB`).
Nu edita `scripts/run_all_models.sh` cât un job `bash scripts/run_all_models.sh`
e RUNNING — bash-ul re-citește fișierul și poate ieși cu `o: command not found`.

Lanțul din 2026-08-31:

| Job | Nume | Ce face | Dependență |
|---|---|---|---|
| 259308 | `romteb` | roster `run_all_models.sh` → `results/` | — |
| 259309 | `romteb-report` | JuRo k, agregare, bootstrap, ploturi în `results/` | `afterok:259308` |
| (nou) | `romteb-legacy` | copiază roster-ul în `legacy/` + ploturi | `afterok:259309` |

Loguri: `logs/slurm-romteb-gpu-<id>.out`, `logs/slurm-romteb-report-<id>.out`,
`logs/slurm-romteb-legacy-<id>.out`.

---

## 1. Doar snapshot `legacy/` + figuri (dacă eval-ul a scris deja JSON-urile)

Fără Slurm, din login, după ce 259308 a terminat:

```bash
./apptainer-exec-romteb.sh python -u scripts/collect_legacy_results.py --copy --out_dir legacy
cp -f results/romteb_ir_bootstrap.csv legacy/ 2>/dev/null || true
./apptainer-exec-romteb.sh scripts/make_plots.sh legacy
```

Figuri: `legacy/plots/`. Roster-ul (fără retriever-e) e în
`scripts/collect_legacy_results.py` (`ROSTER`).

Pe GPU (înlocuiește `JOBID` cu job-ul care trebuie să se termine întâi):

```bash
sbatch --dependency=afterok:JOBID scripts/sbatch_legacy_plots.sh
```

După 259309 (report + bootstrap):

```bash
sbatch --dependency=afterok:259309 scripts/sbatch_legacy_plots.sh
```

Dacă 259309 a picat dar 259308 e COMPLETED:

```bash
sbatch --dependency=afterok:259308 scripts/sbatch_legacy_plots.sh
# sau imediat:
sbatch scripts/sbatch_legacy_plots.sh
```

---

## 2. Report (agregare `results/` + bootstrap + ploturi)

```bash
sbatch --dependency=afterok:JOBID_EVAL scripts/sbatch_post_eval.sh
```

Manual (tot prin Apptainer, nu pe CPU de login pentru bootstrap greu — e OK
și pe login; e CPU):

```bash
bash scripts/post_eval_report.sh results
```

---

## 3. Eval complet (un model roster, GPU)

Nu porni un al doilea `run_all_models.sh` pe același `results/` cât 259308
e RUNNING.

```bash
sbatch --job-name=romteb scripts/run_gpu_job.sh bash scripts/run_all_models.sh results
```

Un singur model / task-uri:

```bash
sbatch --job-name=romteb-gpu scripts/run_gpu_job.sh \
  romteb/run_benchmark.py --model MODEL --loader auto \
  --output results
```

Qwen 0.6B / 8B au ratat task-urile custom cu `loader=auto`. Reluare:

```bash
sbatch --job-name=qwen-st scripts/run_gpu_job.sh \
  romteb/run_benchmark.py --model Qwen/Qwen3-Embedding-0.6B --loader st \
  --output results
```

(la fel pentru `Qwen/Qwen3-Embedding-8B`).

KaLM mini (nu e în job-ul 259308):

```bash
sbatch --job-name=kalm-mini scripts/run_gpu_job.sh \
  romteb/run_benchmark.py \
  --model KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 \
  --loader st --trust_remote_code --output results
```

Arctic a picat la load; retry același pattern cu
`Snowflake/snowflake-arctic-embed-m-v2.0` + `--loader st --trust_remote_code`.

---

## 4. RoSTS (repo Hub 404)

Fără dataset, STS lipsește pentru toate modelele. Doar dacă vrei re-push
(scrie pe Hub și actualizează pin-ul):

```bash
./apptainer-exec-romteb.sh romteb/data_prep/prep_ro_sts.py
```

Apoi re-eval RoSTS pe modelele din roster, de ex.:

```bash
sbatch --job-name=rosts scripts/run_gpu_job.sh \
  romteb/run_benchmark.py --model intfloat/multilingual-e5-base --loader auto \
  --tasks RoSTS --output results
```

---

## 5. Verificări

```bash
squeue -u "$USER" -o '%.18i %.16j %.8T %.10M %.25R'
tail -f logs/slurm-romteb-gpu-259308.out
tail -f logs/slurm-romteb-report-259309.out
ls legacy/plots/
```
