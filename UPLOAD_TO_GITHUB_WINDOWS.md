# Upload to GitHub from Windows CMD

Repository:
https://github.com/prog85sameeh-coder/SRACR-Med

## 1. Clone the repository

```bat
cd C:\Users\Sameeh\Documents
git clone https://github.com/prog85sameeh-coder/SRACR-Med.git
cd SRACR-Med
```

## 2. Copy files

Copy the **contents** of the folder `SRACR_Med_GitHub_READY` into the cloned `SRACR-Med` folder.

## 3. Commit and push

```bat
git status
git add .
git commit -m "Add SRACR-Med reproducibility code and artifacts"
git branch -M main
git push -u origin main
```

If Git asks for your identity:

```bat
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Then repeat the commit/push commands.

## 4. Before public release

- Edit `CITATION.cff.template` with the real paper author list.
- Rename it to `CITATION.cff`.
- If the journal does not permit the manuscript DOCX in a public repository, delete `paper/IJIES_Revised_FINAL_CONDENSED_RED.docx`.
- Confirm that no raw images, passwords, tokens, or local absolute paths were added.

## 5. Create v1.0.0

```bat
git tag -a v1.0.0 -m "SRACR-Med manuscript reproducibility release"
git push origin v1.0.0
```

Then create a GitHub Release for `v1.0.0` and attach the checkpoint/reproducibility ZIP files as Release assets.
