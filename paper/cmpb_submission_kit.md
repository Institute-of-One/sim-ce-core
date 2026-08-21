# CMPB submission kit — IORN-008 (sim_ce_core)

Target: *Computer Methods and Programs in Biomedicine* (Elsevier), **Full Length
Article** (the portal's name for an original research paper).
Submission through Editorial Manager. Single-anonymised review; the editors screen first
and, if suitable, send to a minimum of two reviewers.

**Run `python paper/presubmission_check.py` before building anything to send.** It must
print `ready to submit`.

## Why this venue

Chosen for a structural property rather than for impact. A full recent issue of CMPB
(Vol 272, December 2025) contains 33 research articles, 2 reviews, 2 letters, 2
editorials and 1 correspondence — and **no Technical Note or Short Communication
category**. There is no short-form research bin an editor could reclassify a methods
paper into. By contrast Physica Medica Vol 129 carried 2 Technical Notes and 3 Short
Communications against 19 research articles, and European Radiology Experimental lists
Technical note, Brief report and Methodology article as separate types.

That mattered because the classification, not the correctness, is what has gone wrong
before: JACMP reclassified IORN-004 from Original Article to Technical Note, and JMI desk
rejected two companion papers on framing.

CMPB's aims are also the right shape for this work — "to report new computer
methodologies applied in biomedical areas" and "the eventual distribution of demonstrable
software" — so the software is the entry ticket rather than the demotion trigger. The
readership explicitly includes "(radio)physicists" and "pharmacologists".

**Cost.** CMPB is hybrid. The open-access charge is USD 3,320 excluding tax for every
article type; the subscription route carries none, and no funder here mandates open
access. Choose subscription at submission — it is an initial decision that can be changed
to open access on acceptance, and the reverse is harder.

**What subscription costs instead.** A twelve-month embargo before the accepted
manuscript may go into an institutional or funder repository. Nothing here conflicts with
it: the code is MIT-licensed and public but is not the accepted manuscript, and there is
no preprint to fall foul of it. It does mean the accepted manuscript cannot be archived
on Zenodo beside the code until twelve months after publication. Worth knowing for an
author whose declared position is that code and data should be released openly.

**Honest risk.** CMPB is selective (Impact Factor 6.4, CiteScore 11.9 — higher than
Medical Physics or Physica Medica). This trades reclassification risk for rejection risk.
A rejection leaves the manuscript intact for the next venue, which a demotion does not.

## Requirements verified

- **Abstract** — "concise and factual", **maximum 250 words**, checked at build.
- **Structured headings** — Background and Objectives / Methods / Results / Conclusions.
- **Keywords** — 7.
- **References** — 13, every one carrying a DOI, all cited, all listed, in order of first
  citation. Verified against Crossref records rather than typed.
- **Figures** — 8, each named in the prose, each captioned once, each fitting the text
  block at print size.
- **Multiple corresponding authors** are no longer accepted; there is one author.
- **Ethics statement** — required at submission whether or not the study needs one.

## Editorial Manager field answers

**Article type.** **Full Length Article** — the portal's name for it. The submission
form does not use the phrase "Original Research"; the options are Full Length Article,
Review Article, Correspondence, and two AI Radiomics special-issue variants.

Do not choose either special-issue variant. This paper is not radiomics: it extracts no
texture or shape features. And its finding is that neural estimators do not beat the
classical one in this regime, which is a poor fit for an AI special issue and its guest
editors.

**Title and abstract.** Do not retype them. `python paper/make_portal_fields.py` writes
`paper/build/portal_fields.txt` from the built manuscript, as plain text with the
citation numbers stripped — a bracketed number in an abstract field points at nothing.
The portal's copy of the front matter is the half an editor reads first, and on the
companion submission it went three weeks out of date without anything noticing.

**Corresponding and sole author.** Shuji Yamamoto, Institute of One, LISIT Co., Ltd.,
Tokyo, Japan. yamamoto@lisit.jp. ORCID 0000-0001-9211-1071.

**Ethics declaration.** The portal asks whether the research involved *human (organ,
tissue, cell or participation data)*, animal, or neither. **Answer human.** It does not
ask whether participants were recruited; it asks whether human-derived data were used,
and the external arm is liver CT from twenty patients. Answering "not applicable" would
state that no human data were involved, which the paper's own external validation
contradicts, and an editor who later notices that has been told something false. There
is no cost to declaring: the follow-up is where the absence of required approval is
explained.

The follow-up questions and their answers:

| Question | Answer |
|---|---|
| Complies with relevant laws and guidelines | yes |
| Reviewed by an ethics committee | **No, not required** — not "Yes, and exempt", which would mean a committee reviewed it and granted exemption. No committee saw it |
| Clinical trial | no |
| All participants gave written informed consent | **No** — answering yes would attest that *we* obtained consent. There were no participants in this study |
| Privacy rights observed | yes — the case labels are the archive's own de-identified identifiers |
| Included human biological materials | **No** — that question is about organs, tissue and cells. This study used images. It does not contradict declaring human *data* |
| Case report or series | no |

Guidelines field: Declaration of Helsinki; Ethical Guidelines for Medical and Biological
Research Involving Human Subjects (Japan, MEXT/MHLW/METI); The Cancer Imaging Archive
data use policy (CC BY 4.0). Confirm the national guideline's title against the form you
normally cite — you are the one attesting.

Statement of why review was not required, which is printed in the article:

> This study did not require review by an ethics committee. It recruited no participants
> and collected no data. Its only human-derived material is a publicly available, fully
> de-identified imaging collection used as secondary data: HCC-TACE-Seg, The Cancer
> Imaging Archive, DOI 10.7937/TCIA.5FNA-0924, released under CC BY 4.0. Research using
> only anonymised information from which individuals cannot be identified falls outside
> the scope requiring institutional review under the applicable national guidelines.
> Institutional approval and participant consent for the original collection are held by
> the contributing institution, and no additional approval was sought or required for
> this secondary use.

"Not required" on its own reads as a study nobody approved. Naming who holds the approval
is what makes it a statement rather than an absence.

Approval and consent:

> Secondary analysis of a publicly available, fully de-identified imaging collection:
> HCC-TACE-Seg, The Cancer Imaging Archive, DOI 10.7937/TCIA.5FNA-0924, released under
> CC BY 4.0. No participants were recruited and no data were collected for this study.
> Institutional approval and participant consent are held by the original collection; no
> additional approval was required for this secondary use of de-identified public data.

The portal generates a statements file from these answers and prints it in the published
article. Download and read it before ticking the confirmation: it is a signature.

**Declaration of competing interests.** The author is Representative Director (CEO) of
LISIT Co., Ltd. and Chief Executive Officer of TexelCraft OÜ; Institute of One is the
open-research initiative of LISIT Co., Ltd. Neither company sells or licenses any product
related to the subject of this manuscript, and the work used no client or patient data.
Two further interests bear on the subject rather than on finance: the software the study
runs on was written by the author, who is therefore both the implementer and the
assessor, as the Limitations state; and the manuscript argues for openly referenced
validation while the author's research initiative is founded on releasing code and data
openly. There are no other competing interests.

**Funding.** This research received no external funding.

**Declaration of generative AI use.** Generative AI (Claude, Anthropic, through the
Claude Code command-line tool) was used in preparing this work: scaffolding and
refactoring the software, drafting tests, writing the figure and analysis scripts, and
drafting and revising manuscript prose. It was not used to design the study, to choose
the endpoints, or to decide what the results mean. No numerical result came from the
model — every number is emitted by executed code into machine-readable files and resolved
into the text at build time, and the test suite fails if the two disagree. The author
designed the study, re-executed every result, and is solely accountable. No AI system is
an author. This is stated in the manuscript itself, not only here.

**Data statement.** Data available — the code, the frozen results and the commands that
regenerate every figure are in the repository under the MIT licence. The external imaging
is the HCC-TACE-Seg collection in The Cancer Imaging Archive, DOI
10.7937/TCIA.5FNA-0924, CC BY 4.0.

**Preprint.** **Opt in to the journal's free SSRN service.** arXiv requires an endorser
the author does not have and medRxiv does not recognise the affiliation as a research
institution, so this route is the only one open — and it bypasses both, with no endorser
and no institutional gate.

It posts only after the manuscript passes initial desk review, so a desk rejection leaves
nothing public. It carries a preprint DOI and links to the version of record on
publication. And it offsets what the subscription route costs: the twelve-month embargo
on depositing the accepted manuscript binds much less when the submitted version is
already readable.

The cover letter says so. "No preprint of this work exists" would have been true at the
moment of submission, since SSRN posts later, but an author who writes that while opting
into the journal's own preprint service creates a state that needs explaining, and not
creating it is cheaper than explaining it.

**Previously published.** No part of this manuscript has been published or submitted
elsewhere.

## Suggested reviewers

**Do not generate names and email addresses.** Fabricated reviewer contacts are the
mechanism of a documented peer-review fraud, and a plausible-looking address for a real
person is worse than no suggestion. If Editorial Manager requires suggestions, take the
names from the reference list — the authors of references 3, 5, 9, 10 and 11 work
directly on this problem — and obtain each address from the corresponding-author line of
a published paper or an institutional page, checked at the time of submission.

Note that references 3, 5, 6 and 9 are cited in the manuscript. Citing a prospective
reviewer does not disqualify them; flag it if one is proposed.

## Files to upload

| File | Built by |
|---|---|
| `paper/build/manuscript.docx` | `python paper/build_docx.py` |
| `paper/cover_letter_cmpb.txt` | maintained here |

Figures are embedded in the manuscript at the position the prose names them. If Editorial
Manager requires them separately, they are in `paper/figures/` as
`fig1_…` through `fig8_…`, and their captions are in `paper/README.md`.

## What this paper claims, and what it does not

Stated here so the cover letter and the manuscript cannot drift apart.

**Claimed.** An exact scale symmetry of the reduced Bae model that no sampling density
breaks. The Cramér–Rao bound on physiological recovery at routine phase counts. That the
bound predicts measured recovery error across designs and estimators. That a closed-form
fit attains it while physics-informed and amortized estimators run at 1.5 and 1.7 times
it. That a pre-contrast acquisition carries no information and an arterial phase carries
3.5 times what a portal venous phase does.

**Not claimed.** Novelty for applying physics-informed networks to contrast kinetics —
that is prior art, cited. Novelty for Fisher-information-based experimental design — that
is textbook, cited. Any statement about parameter recovery on real data, where there is
no ground truth. Any generalisation of the scale symmetry beyond this reduced model.
