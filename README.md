# Image-based Profiling Template

This repository was derived from a [template repository](https://github.blog/2019-06-06-generate-new-repositories-with-repository-templates/) located at https://github.com/cytomining/profiling-template.
The purpose of the repository is to weld together a versioned data processing pipeline with versioned processed output data for a single image-based profiling experiment.

(Derived from this [template](https://github.com/broadinstitute/pooled-cell-painting-profiling-template))

**DO THE FOLLOWING AFTER GENERATING A NEW REPO:**

- Change the title of the README to **Image-based Profiling for [ProjectName]\(project-url\)**.
- Delete the **Setup** section
- Keep the notes section, and edit as necessary.
- Keep the metadata section if you are adopting it, and edit as necessary.

## Setup

To correctly initialize the repository, we need to perform several manual steps.

### Step 0: Create a new repository **using this repository as a template**

By spinning up a new repo using this repo as a template, you will retain all code, configuration files, computational environments, and directory structure that a standard image-based profiling workflow expects and produces.

### Step 1: Fork the `profiling-recipe` repo

We first want to [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the official profiling recipe located at https://github.com/cytomining/profiling-recipe.

* **Result:** The fork creates a copy of a recipe repository.
* **Goals:** 1) Remove the connection to official recipe updates to avoid unintended weld versioning reversal; 2) Enable independent updates to fork code that does not impact official recipe.
* **Execution:** See [forking instructions](https://help.github.com/en/github/getting-started-with-github/fork-a-repo).


### Step 2: Create a submodule inside this repository of the forked recipe

Next, we will create a [submodule](https://gist.github.com/gitaarik/8735255) in this repo.

* **Result:** Adding a submodule initiates the weld.
* **Goals:** 1) Link the processing code (recipe) with the data (current repo); 2) Require a manual step to update the recipe to enable asynchronous development.
* **Execution:** See below

```bash
# In your terminal, clone the repository you just created (THIS REPO)
USER="INSERT-USERNAME-HERE"
REPO="INSERT-NAME-HERE"
git clone git@github.com:$USER/$REPO.git

# Navigate to this directory
cd $REPO

# Add the recipe submodule
git submodule add https://github.com/$USER/profiling-recipe.git profiling-recipe
```

Refer to ["Adding a submodule"](https://gist.github.com/gitaarik/8735255#adding-a-submodule) for more details.

### Step 3: Commit the submodule

Lastly, we will [commit](https://help.github.com/en/desktop/contributing-to-projects/committing-and-reviewing-changes-to-your-project#about-commits) the submodule to github.

* **Result:** Committing this change finalizes the weld.
* **Goals:** 1) Track the submodule (recipe) version with the current repository.
* **Execution:** See below

```bash
# Add, commit, and push the submodule contents
git add profiling-recipe
git add .gitmodules
git commit -m 'finalizing the recipe weld'
git push
```

### Step 4: Fill out experiment information

This step ensures that relevant experimental information is captured.
We recommend all users, both internal and external, to diligently track project metadata. For those without a system, a template is provided below.

## Metadata template

Fill out the whole template on the first batch; for subsequent batches only add what is different.

<details>
    
```
Cell type : _______ (ex: U2OS)
Cell source: ________ (ex: Collab lab) (ex: GPP)
Plate size : _______ (ex: 384)
Plate brand : _______ (ex: Cell carrier Ultra)
Cell densit(y/ies) : _______ (ex: 2K/well) (ex: Columns A-D 1K/well, Columns G-H 500/well)
Type of perturbation : ___________ (ex: Gene overexpression) (ex: CRISPR KO + compounds)
If (at least partially) genetic: 
    Genetic introduction method : __________ (ex : lentiviral transduction)
    Selection method : _________ (ex: None) (ex: Puromycin 1ug/mL 24 hrs)
    Number of genes : __________ (ex: 384)
    Number of perturbations per gene : __________ (ex: N/A) (ex: 4 guides per gene)
If (at least partially) chemical :
    Number of chemicals : __________ (ex: 384)
    Number of dose points per chemical : _________ (ex: 1)
Number of replicates per perturbation : __________ (ex: 5)
Perturbation time point : _____________ (ex: 72 hrs)
Staining protocol : ____________ (ex: CellPainting v3 (LINK)) (ex: LipocytePainting (CITATION)) (ex: 1:500 gt anti tubulin (cat #), 1:1000 A488 anti gt (cat #))
Microscope : ________ (ex: Opera Phenix ) 
Mode : ________ (ex: Confocal) 
Excitation / emission details : ______________ (ex: ex 488 laser, em 550/50; ex 568 laser, em 600/30) (ex: see Index.idx.xml file)
Objective : _____________ (ex: 20X water 1.0NA)
Binning : ____________ (ex: 1x1)
Sites per well : __________ (ex: 9)
Pixel size : ____________ (ex: 0.656um)
Number of Z planes : _______ (ex: 3)
Z plane spacing : ________ (ex: 1um)
```

</details>

## Notes

To download the data in this repo, first, clone this repo

```bash
git clone git@github.com:<org>/<repo>.git
```

then, download the profiles

```bash
cd <repo>
dvc pull
```

### AWS configuration

The DVC cache is typically stored in an AWS S3 bucket, so you will need run `aws configure` before running `dvc pull`.

If the DVC location is not publicly accessible, you will need AWS credentials to access it.

If the DVC location is not publicly accessible, to access the files stored via DVC, you will need to created a IAM user with the `AmazonS3ReadOnlyAccess` policy attached:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:Get*",
                "s3:List*",
                "s3-object-lambda:Get*",
                "s3-object-lambda:List*"
            ],
            "Resource": "*"
        }
    ]
}
```
