# mthm503

![CI](https://github.com/<YOUR USER NAME>/<YOUR REPOSITORY NAME>/actions/workflows/main.yml/badge.svg)

This repository contains the materials for the course **MTHM503: Applications of Data Science and Statistics**.

You will use *your own copy* of this repository (created via GitHub Classroom or by forking) for all assessed and non-assessed work on this module.

> **Note**: In a professional setting, a DevOps engineer (or team of engineers) would be responsible for setting up infrastructure like this.
> In this course, you only need to operate *within* the provided infrastructure.

---

## How you should work in this repository

- You should **always do your development work on the `main` branch**
- Automated tests and checks run on `main`
- For **formative feedback**, open a pull request from:
  - **compare**: `main`
  - **base**: `feedback`

Do **not** merge feedback pull requests — tutors will comment on them and then close them.

(Full details are given below.)

---

## Setting up for the first time

The recommended development environment for this module is **Visual Studio Code (VS Code)**. More details:  [VS_CODE.md](VS_CODE.md)

I have tested the setup on University Virtual Machines, University laptops and lab machines, and personal Linux machines. It *does* work on Windows, but you will need to follow the instructions carefully.

The **single most important instruction** is:

> **Run commands from the command line (Anaconda Prompt / terminal) inside your repository directory.**

Setup steps:

- Make sure you have a GitHub account and that it is registered as a *student account* with GitHub. Further instructions are available in ELE.
- Open VS Code and clone your repository, or use **File → Open Folder** if you have already cloned it.
- Open **Anaconda Prompt** (Windows) or a terminal (macOS/Linux) and navigate to the repository directory you just downloaded.
- From that directory, run:
  ```sh
  conda env create -f environment.yml
  ```
- Activate the environment:
  ```sh
  activate python-exercises
  ```
- In VS Code, select this environment as your Python interpreter when prompted.

⚠️ You need to complete this setup in **Week 1**. Many things will not run correctly until it is done, and you will not be able to submit coursework otherwise.

---

## Developing reproducible pipelines

The core working method in this module is:

> Write functions that pass unit tests, and integrate them into reproducible data pipelines.

Key points:

1. You can work **interactively in Python** using VS Code (Python Interactive Window, terminal REPL, or notebooks).
2. You can also run Python scripts directly from the command line.
3. Use interactive experimentation to develop ideas, then convert them into functions and scripts.
4. Reproducible pipelines are managed using **DVC**. You will typically reproduce pipeline stages using:
   ```sh
   dvc repro
   ```

All intermediate data products are tracked and managed by DVC, allowing pipelines to be re-run reliably and inspected at any stage.

As the course progresses, you are encouraged — especially for higher marks — to write **your own tests** and extend the pipelines.

---

## Formative feedback (pull requests)

Formative feedback is given via **pull requests against the `feedback` branch**.

### How to request feedback

1. Do all your work on `main`
2. When you want feedback:
   - Open a pull request
   - **Base branch**: `feedback`
   - **Compare branch**: `main`
3. Select the appropriate pull request template (classification, regression, reporting, etc.)

Tutors will leave comments on the pull request and then **close it**.

> **Do not merge feedback pull requests.** They exist only to support written feedback and discussion.

```
        +----------------------------+
        | 1. Get your repository    |
        |    (GitHub Classroom)     |
        +----------------------------+
                   |
                   v
        +----------------------------+
        | 2. Do the work on `main`  |
        +----------------------------+
                   |
                   v
        +----------------------------+
        | 3. Open pull request      |
        |    main → feedback        |
        |    Select correct         |
        |    PR template            |
        +----------------------------+
                   |
                   v
        +----------------------------+
        | 4. Tutors review          |
        |    and comment            |
        +----------------------------+
```
