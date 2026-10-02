#!/usr/bin/env bash
# Regenerates every member notebook from the single source in scripts/nb_src/.
# Only the MEMBER line differs between members. Run from the repo root:  bash scripts/build_notebooks.sh
# (Only needed if you edit a file in scripts/nb_src/. Executed notebooks are kept unless FORCE=1.)
set -euo pipefail
command -v jupytext >/dev/null || pip install -q jupytext
for member in ruchika_chaurasia member_b; do
  for nb in task1_llm task2_sentiment task3_gan; do
    sed "s/^MEMBER = \"ruchika_chaurasia\"/MEMBER = \"${member}\"/" "scripts/nb_src/${nb}.py" > "/tmp/${nb}_${member}.py"
    target="${nb}/${member}/src/${nb}.ipynb"
    # never overwrite an EXECUTED notebook (it holds submitted results) unless FORCE=1
    if [ -f "$target" ] && grep -q '"output_type"' "$target" && [ "${FORCE:-0}" != "1" ]; then
      echo "keeping executed $target"
      [ "$nb" = "task3_gan" ] && jupytext --quiet --to ipynb "/tmp/${nb}_${member}.py" -o "${nb}/${member}/src/task3_gan_v2.ipynb" && echo "  -> wrote ${nb}/${member}/src/task3_gan_v2.ipynb for extra runs"
      continue
    fi
    jupytext --quiet --to ipynb "/tmp/${nb}_${member}.py" -o "$target"
  done
  cp scripts/nb_src/cyclegan_models.py "task3_gan/${member}/src/cyclegan_models.py"
  cp scripts/nb_src/evaluate_local.py "task3_gan/${member}/evaluate_local.py"
done
echo "notebooks rebuilt"
