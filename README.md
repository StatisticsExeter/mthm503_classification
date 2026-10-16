# MTHM503: Classification Pipelines

https://github.com/StatisticsExeter/mthm503_classification/actions/workflows/main.yml/badge.svg

This repository contains the coursework and laboratory materials for **MTHM503: Applications of Data Science and Statistics**.

The module focuses on building reproducible machine learning workflows using:

- Python
- Git and GitHub
- Conda
- DVC
- Unit testing with `pytest`
- Automated checking with GitHub Actions
- Quarto reports

You will work in your own copy of this repository throughout the module.

---

# Getting Started

The recommended development environment is **Visual Studio Code**.

> Run commands from a terminal inside the repository directory.

## Create the environment

From the repository root:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate pycw
```

## Verify your setup

Run:

```bash
./run check
```

You should see a series of green ticks confirming:

- Correct conda environment
- Correct Python version
- `pytest` installed
- Quarto available

If any checks fail, resolve those issues before continuing.

---

# Working on the Coursework

There are two classification pipelines:

- Supervised Classification
- Unsupervised Classification

Each pipeline is managed using DVC.

## Running pipeline stages

To run a complete supervised workflow:

```bash
dvc repro supervised_report
```

To run a complete unsupervised workflow:

```bash
dvc repro unsupervised_report
```

You can also run intermediate stages if you wish:

```bash
dvc repro supervised_model
```

```bash
dvc repro unsupervised_cluster
```

DVC will automatically run any prerequisite stages that are required.

---

# Laboratory Mode and Coursework Mode

The repository supports two modes:

```yaml
mode: labs
```

and

```yaml
mode: cw
```

During laboratory sessions you should normally use:

```yaml
mode: labs
```

When you begin the coursework, switch to:

```yaml
mode: cw
```

by editing `params.yaml`.

You will also need to select the appropriate coursework configuration in the `active:` section.

For example:

```yaml
active:
  supervised_target_variable: "built_age"
```

Use the configuration specified in the coursework brief.

---

# Running Checks

Before committing work, run the appropriate checks locally.

## Supervised Classification

```bash
./run super
```

## Unsupervised Classification

```bash
./run unsuper
```

## All Checks

```bash
./run all
```

These commands will:

1. Check Python syntax.
2. Check for major coding errors.
3. Run unit tests.
4. Run linting checks.

## Verbose Output

For more detailed test output:

## Verbose Output

For more detailed test output:

```bash
./run super verbose
```

```bash
./run unsuper verbose
```

```bash
./run all verbose
```



# GitHub Actions

Whenever you push changes:

```bash
git push
```

GitHub Actions will automatically run the same checks used by the local `run` script.

A successful GitHub Actions run indicates that:

- Your code runs on a clean machine.
- Unit tests pass.
- Linting checks pass.

You should ensure that GitHub Actions is passing before submission.



# Formative Feedback

Formative feedback will be provided through Moodle.

When requesting feedback, provide:

- Repository URL
- Commit hash

For example:

```text
Repository:
https://github.com/username/mthm503-classification

Commit:
a43f0b8
```

This allows tutors to review the exact version of the work you wish to discuss.

---

# Typical Workflow

```text
1. Write code

2. Run checks
   ./run super
   ./run unsuper

3. Run pipeline
   dvc repro supervised_report
   dvc repro unsupervised_report

4. Commit changes
   git add .
   git commit -m "Implement classifier"

5. Push to GitHub
   git push

6. Confirm GitHub Actions passes
```

---

# Getting Help

If something is not working:

1. Run:

   ```bash
   ./run check
   ```

2. Read the error message carefully.

3. Consult the module discussion forum.

4. Bring specific error messages to labs or office hours.

When asking for help, include the complete error message whenever possible. This makes problems much easier to diagnose.

---

# Academic Integrity

All submitted work must be your own.

You are encouraged to discuss ideas with other students, but the code, analysis and report that you submit must represent your own work.

Where external resources have been used, they should be acknowledged appropriately.
