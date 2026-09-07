#!/usr/bin/env bash
# Build the thesis with XeLaTeX + BibTeX inside a pinned TeX Live container.
#
# No LaTeX installation is required on the host; only Docker. The container is
# run as the calling user so that generated files stay writable.
#
# Usage:
#   ./build.sh              full report (thesis.tex → thesis.pdf)
#   ./build.sh short        concise report (thesis-short.tex → thesis-short.pdf)
#   ./build.sh all          full report then concise report
#   ./build.sh quick        single xelatex pass of the full report
#   ./build.sh short-quick  single xelatex pass of the concise report
#   ./build.sh clean        remove auxiliary files

set -euo pipefail

IMAGE="${TEXLIVE_IMAGE:-ghcr.io/xu-cheng/texlive-full:latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"

run_tex() {
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$HERE":/work \
    -w /work \
    "$IMAGE" \
    sh -c "$1"
}

summarize() {
  local job="$1"
  local log="$2"
  echo "=== ${job} ==="
  echo "--- bibtex ---"
  if [[ -f bibtex-${job}.log ]]; then
    grep -iE "error|warning|repeated" "bibtex-${job}.log" | head -10 || echo "(clean)"
  elif [[ -f bibtex.log && "${job}" == "thesis" ]]; then
    grep -iE "error|warning|repeated" bibtex.log | head -10 || echo "(clean)"
  else
    echo "(no bibtex log)"
  fi
  echo "--- errors ---"
  grep -n '^!' "${log}" || echo "(none)"
  echo "--- undefined references / citations ---"
  grep -nE 'LaTeX Warning: (Reference|Citation) .* undefined' "${log}" | head -20 || true
  echo "--- worst overfull boxes ---"
  if grep -qE 'Overfull \\hbox \([0-9]+\.[0-9]+pt' "${log}"; then
    grep -oE 'Overfull \\hbox \([0-9]+\.[0-9]+pt' "${log}" \
      | sed -E 's/.*\(([0-9.]+)pt/\1/' | sort -rn \
      | awk 'NR==1{printf "worst=%spt  ", $1} $1>15{n++} END{printf "count>15pt=%d\n", n+0}'
  else
    echo "worst=0pt  count>15pt=0"
  fi
  echo "--- result ---"
  grep -oE "Output written on ${job}.pdf \\([0-9]+ pages" "${log}" | tail -1 || echo "no PDF line"
  ls -la "${job}.pdf" 2>/dev/null || echo "${job}.pdf missing"
}

build_job() {
  local job="$1"
  local pass_mode="$2"
  if [[ "${pass_mode}" == "quick" ]]; then
    run_tex "xelatex -interaction=nonstopmode ${job}.tex > ${job}-build.log 2>&1 || true"
  else
    run_tex "
      set -e
      xelatex -interaction=nonstopmode ${job}.tex > ${job}-pass1.log 2>&1 || true
      bibtex ${job} > bibtex-${job}.log 2>&1 || true
      xelatex -interaction=nonstopmode ${job}.tex > ${job}-pass2.log 2>&1 || true
      xelatex -interaction=nonstopmode ${job}.tex > ${job}-build.log 2>&1 || true
    "
  fi
  cd "$HERE"
  summarize "${job}" "${job}-build.log"
}

case "$MODE" in
  clean)
    cd "$HERE"
    rm -f thesis.aux thesis.bbl thesis.blg thesis.log thesis.lof thesis.lot \
          thesis.out thesis.toc thesis.fls thesis.fdb_latexmk \
          build.log pass1.log pass2.log bibtex.log
    rm -f thesis-short.aux thesis-short.bbl thesis-short.blg thesis-short.log \
          thesis-short.out thesis-short.toc thesis-short.lof thesis-short.lot \
          thesis-short-build.log thesis-short-pass1.log thesis-short-pass2.log \
          thesis-build.log thesis-pass1.log thesis-pass2.log \
          bibtex-thesis.log bibtex-thesis-short.log
    rm -f chapters/*.aux chapters/short/*.aux front/*.aux front/template/*.aux bibs/*.aux
    echo "cleaned"
    ;;
  quick)
    build_job thesis quick
    ;;
  short-quick)
    build_job thesis-short quick
    ;;
  full)
    build_job thesis full
    # Keep the historical log names so existing notes still apply.
    cd "$HERE"
    if [[ -f thesis-build.log ]]; then
      cp -f thesis-build.log build.log
      cp -f thesis-pass1.log pass1.log 2>/dev/null || true
      cp -f thesis-pass2.log pass2.log 2>/dev/null || true
      cp -f bibtex-thesis.log bibtex.log 2>/dev/null || true
    fi
    ;;
  short)
    build_job thesis-short full
    ;;
  all)
    build_job thesis full
    cd "$HERE"
    if [[ -f thesis-build.log ]]; then
      cp -f thesis-build.log build.log
      cp -f thesis-pass1.log pass1.log 2>/dev/null || true
      cp -f thesis-pass2.log pass2.log 2>/dev/null || true
      cp -f bibtex-thesis.log bibtex.log 2>/dev/null || true
    fi
    build_job thesis-short full
    ;;
  *)
    echo "usage: $0 [full|short|all|quick|short-quick|clean]" >&2
    exit 2
    ;;
esac
