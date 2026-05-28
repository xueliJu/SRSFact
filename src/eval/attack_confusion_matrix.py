import argparse
import json
import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)


CLASSES = ["SUPPORTED", "NEI", "REFUTED", "CONFLICTING"]

LABEL_ALIASES = {
    "supported": "SUPPORTED",
    "not enough information": "NEI",
    "nei": "NEI",
    "refuted": "REFUTED",
    "conflicting evidence": "CONFLICTING",
    "conflicting": "CONFLICTING",
    "conflicting evidence/cherrypicking": "CONFLICTING",
    "cherry-picking": "CONFLICTING",
    "cherry picking": "CONFLICTING",
    "error: refused to answer": "REFUSED_TO_ANSWER",
    "refused to answer": "REFUSED_TO_ANSWER",
}


def parse_label(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"Label must be a string, got: {type(raw)}")

    normalized = raw.strip().lower()
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]

    upper = raw.strip().upper()
    if upper in CLASSES or upper == "REFUSED_TO_ANSWER":
        return upper

    raise ValueError(f"Unsupported label string: {raw}")


def heatmap(data, row_labels, col_labels, show_cbar=True, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    im = ax.imshow(data, **kwargs)

    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.ma.is_masked(data[i, j]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j]), **kw)
                texts.append(text)

    return texts


def plot_confusion_matrix(predictions: list[str],
                          ground_truth: list[str],
                          classes: list[str],
                          benchmark_name: str,
                          output_dir: Path):
    class_conversion = {c: v for v, c in enumerate(classes)}

    confusion_matrix = np.zeros((len(classes), len(classes)), dtype="float")
    for pred, gt in zip(predictions, ground_truth):
        if pred == "REFUSED_TO_ANSWER":
            continue
        if pred not in class_conversion or gt not in class_conversion:
            continue
        confusion_matrix[class_conversion[gt], class_conversion[pred]] += 1

    correct = np.copy(confusion_matrix)
    wrong = np.copy(confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            if i == j:
                wrong[i, j] = np.nan
            else:
                correct[i, j] = np.nan

    fig, ax = plt.subplots()
    v_max = np.max(len(ground_truth) // 3)
    hm, _ = heatmap(correct, classes, classes, cmap="Greens", show_cbar=False,
                    ax=ax, vmin=0, vmax=v_max)
    annotate_heatmap(hm, valfmt="{x:.0f}")
    hm, _ = heatmap(wrong, classes, classes, cmap="Reds", show_cbar=False,
                    ax=ax, vmin=0, vmax=v_max)
    annotate_heatmap(hm, valfmt="{x:.0f}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{benchmark_name} Confusion Matrix")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{benchmark_name}.pdf")
    plt.savefig(output_dir / f"{benchmark_name}.png")
    plt.close(fig)


def read_attack_jsonl(jsonl_path: Path) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    ground_truth: list[str] = []

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skip malformed json at %s line %d: %s", jsonl_path, line_no, e)
                continue

            if "gt_label" not in record or "pred_label" not in record:
                logger.warning(
                    "Skip incomplete record at %s line %d: missing gt_label/pred_label",
                    jsonl_path,
                    line_no,
                )
                continue

            try:
                gt = parse_label(record["gt_label"])
                pred = parse_label(record["pred_label"])
            except ValueError as e:
                logger.warning("Skip record at %s line %d: %s", jsonl_path, line_no, e)
                continue

            ground_truth.append(gt)
            predictions.append(pred)

    return predictions, ground_truth


def iter_attack_result_jsonl(attack_results_dir: Path):
    for attack_dir in sorted(p for p in attack_results_dir.iterdir() if p.is_dir()):
        results_dir = attack_dir / "results"
        if not results_dir.exists():
            logger.info("Skip %s: no results dir", attack_dir.name)
            continue

        jsonl_files = sorted(results_dir.glob("attack_results_*.jsonl"))
        if not jsonl_files:
            logger.info("Skip %s: no attack_results_*.jsonl", attack_dir.name)
            continue

        if len(jsonl_files) > 1:
            logger.warning(
                "%s has multiple jsonl files, using first one: %s",
                attack_dir.name,
                jsonl_files[0].name,
            )

        yield attack_dir.name, jsonl_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Batch plot confusion matrices for all attack results."
    )
    parser.add_argument(
        "--attack-results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "attack" / "attack_results",
        help="Root directory containing attack result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "attconfusion",
        help="Directory to save confusion matrix images.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    attack_results_dir = args.attack_results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not attack_results_dir.exists():
        raise FileNotFoundError(f"attack_results dir not found: {attack_results_dir}")

    done = 0
    for attack_name, jsonl_path in iter_attack_result_jsonl(attack_results_dir):
        predictions, ground_truth = read_attack_jsonl(jsonl_path)
        if not predictions:
            logger.warning("Skip %s: no valid records", attack_name)
            continue

        plot_confusion_matrix(
            predictions=predictions,
            ground_truth=ground_truth,
            classes=CLASSES,
            benchmark_name=attack_name,
            output_dir=output_dir,
        )
        done += 1
        logger.info(
            "Saved confusion matrix for %s -> %s/%s.[png/pdf]",
            attack_name,
            output_dir,
            attack_name,
        )

    logger.info("Finished. Generated confusion matrices for %d folders.", done)


if __name__ == "__main__":
    main()
