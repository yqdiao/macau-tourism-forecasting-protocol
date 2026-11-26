# S5 history-panel rerun note

`build_panels.build_history_panels` was expanded to join GL03/LL01/NL01/lag_llama/transformer/sarima history caches that already existed on disk but were not joined into S5. After the join, S5 covers 14 configs × Clean-180 / Clean-365 / Full-with-COVID. See `NAN_FILL_FINAL_REPORT.md` and `nan_inventory.md`.
