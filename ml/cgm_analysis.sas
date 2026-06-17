/*============================================================================
  cgm_analysis.sas
  ----------------
  Descriptive and inferential analysis of continuous glucose monitoring (CGM)
  data from the AZT1D 2025 cohort (25 Type 1 Diabetes patients, ~38 days each).

  Demonstrates:
    PROC IMPORT    -- reading CSV into a SAS dataset
    PROC FORMAT    -- labeling clinical glucose zones
    PROC MEANS     -- per-subject descriptive statistics (TIR, mean, CV)
    PROC FREQ      -- frequency of glucose zone categories
    PROC UNIVARIATE-- full distribution analysis with normality tests
    PROC TTEST     -- two-sample comparison (ADA goal met vs. not met)
    PROC CORR      -- correlation between clinical metrics
    PROC REG       -- linear regression (mean glucose ~ basal rate)
    PROC LOGISTIC  -- logistic regression (hypo risk)
    PROC SGPLOT    -- glucose time-series and distribution visualizations
    PROC SGPANEL   -- subject-level faceted glucose distribution
    ODS output     -- HTML report generation

  Data: run python ml/generate_synthetic_data.py first to create
        sample_data/CGM Records/Subject N/synthetic.csv files, then
        prepare the SAS-ready flat file with python tableau/export_for_tableau.py
        (output: tableau/patient_summary.csv and tableau/glucose_timeseries.csv)

  Usage:
    sas cgm_analysis.sas
    or submit in SAS Studio / SAS OnDemand for Academics (free).

  Author:  Genelle Jenkins
  Dataset: AZT1D 2025 — Arizona State University
  Note:    Educational analysis; simulated/anonymised data only.
============================================================================*/

/* -------------------------------------------------------------------------
   0.  Setup & Options
   --------------------------------------------------------------------- */
options nodate nonumber ls=120 ps=max;
ods graphics on / width=800px height=500px;

%let data_dir  = &SYSSCP = WIN ? C:\portfolio\azt1d\tableau : ~/portfolio/azt1d/tableau;
%let out_dir   = ./sas_output;

/* Create output directory if running in a Unix environment */
%macro mkdir(dir);
  %if &SYSSCP ne WIN %then %do;
    systask command "mkdir -p &dir";
  %end;
%mend;
%mkdir(&out_dir);

/* Open HTML report */
ods html file="&out_dir/cgm_clinical_report.html"
         style=HTMLBlue
         title="AZT1D CGM Cohort — Clinical Statistics Report";


/* =========================================================================
   1.  IMPORT  patient_summary.csv  (one row per subject)
   ======================================================================= */
proc import datafile="tableau/patient_summary.csv"
            out=work.patient_summary
            dbms=csv
            replace;
    getnames=yes;
    guessingrows=50;
run;

/* =========================================================================
   2.  IMPORT  glucose_timeseries.csv  (long-format CGM readings)
   ======================================================================= */
proc import datafile="tableau/glucose_timeseries.csv"
            out=work.glucose_ts
            dbms=csv
            replace;
    getnames=yes;
    guessingrows=200;
run;

/* =========================================================================
   3.  FORMATS  — clinical glucose zone labels and risk tier
   ======================================================================= */
proc format;
    value $risk_fmt
        'Low'      = 'Low Risk'
        'Moderate' = 'Moderate Risk'
        'High'     = 'High Risk';

    value $zone_fmt
        'Severe Hypo (<54)'   = '1 Severe Hypo <54 mg/dL'
        'Hypo (54-70)'        = '2 Hypoglycemia 54-70 mg/dL'
        'In Range (70-180)'   = '3 Target Range 70-180 mg/dL'
        'Hyper (180-250)'     = '4 Hyperglycemia 180-250 mg/dL'
        'Severe Hyper (>250)' = '5 Severe Hyper >250 mg/dL';

    value tir_fmt
        low  -< 70  = 'Below ADA Goal (<70%)'
        70  -  100  = 'Meets ADA Goal (>=70%)';

    value hypo_fmt
        low  -< 4   = 'Safe (<4%)'
        4    -  high= 'Unsafe (>=4%)';
run;

/* =========================================================================
   4.  DERIVE  additional clinical indicators
   ======================================================================= */
data work.patient_analysis;
    set work.patient_summary;

    /* ADA goal attainment flags */
    meets_tir_goal_num  = (time_in_range_pct >= 70);
    meets_hypo_goal_num = (time_below_70_pct  <  4);
    meets_cv_goal_num   = (cv_pct             < 36);

    /* Composite flag: patient meeting ALL three ADA targets */
    fully_controlled = (meets_tir_goal_num and meets_hypo_goal_num and meets_cv_goal_num);

    /* Hypoglycemic event rate per day */
    days_observed = n_readings / 288;           /* 5-min intervals → 288/day */
    hypo_events_per_day = n_hypo_events / max(days_observed, 1);

    label
        time_in_range_pct   = 'Time in Range (70-180 mg/dL) %'
        time_below_70_pct   = 'Time Below 70 mg/dL %'
        time_above_180_pct  = 'Time Above 180 mg/dL %'
        time_below_54_pct   = 'Time Below 54 mg/dL % (Severe)'
        cv_pct              = 'Glucose CV %'
        mean_glucose        = 'Mean Glucose (mg/dL)'
        n_hypo_events       = 'N Hypoglycemic Events'
        risk_tier           = 'Hypoglycemia Risk Tier'
        fully_controlled    = 'Meets All 3 ADA Targets'
        hypo_events_per_day = 'Hypo Events per Day';

    format risk_tier $risk_fmt.;
run;


/* =========================================================================
   5.  PROC MEANS  — cohort-level descriptive statistics
   ======================================================================= */
title "Table 1. Cohort Glycemic Control — Descriptive Statistics";
proc means data=work.patient_analysis
           n mean std median min max q1 q3 maxdec=2;
    var time_in_range_pct
        time_below_70_pct
        time_above_180_pct
        time_below_54_pct
        cv_pct
        mean_glucose
        n_hypo_events
        hypo_events_per_day;
run;
title;

/* Per-risk-tier breakdown */
title "Table 2. Glycemic Metrics by Risk Tier";
proc means data=work.patient_analysis
           n mean std median maxdec=2;
    class risk_tier;
    var time_in_range_pct time_below_70_pct cv_pct n_hypo_events;
    format risk_tier $risk_fmt.;
run;
title;


/* =========================================================================
   6.  PROC FREQ  — ADA goal attainment counts
   ======================================================================= */
title "Table 3. ADA Goal Attainment Across Cohort";
proc freq data=work.patient_analysis;
    tables meets_tir_goal_num
           meets_hypo_goal_num
           meets_cv_goal_num
           fully_controlled
           risk_tier / nocum nopercent;
    format meets_tir_goal_num tir_fmt.
           meets_hypo_goal_num hypo_fmt.;
run;
title;

/* Cross-tabulation: TIR goal × risk tier */
title "Table 4. TIR Goal Attainment by Risk Tier";
proc freq data=work.patient_analysis;
    tables risk_tier * meets_tir_goal_num / chisq expected nocol;
    format meets_tir_goal_num tir_fmt. risk_tier $risk_fmt.;
run;
title;


/* =========================================================================
   7.  PROC UNIVARIATE  — distribution of TIR with normality test
   ======================================================================= */
title "Figure 1. Distribution of Time-in-Range (TIR %)";
proc univariate data=work.patient_analysis normal;
    var time_in_range_pct cv_pct n_hypo_events;
    histogram time_in_range_pct / normal kernel;
    inset n mean std skewness kurtosis / position=ne;
    probplot time_in_range_pct / normal(mu=est sigma=est);
    label time_in_range_pct='TIR %';
run;
title;


/* =========================================================================
   8.  PROC TTEST  — compare TIR in patients meeting vs. not meeting ADA goal
   ======================================================================= */
title "Table 5. Comparison: Patients Meeting vs. Not Meeting TIR Goal";
proc ttest data=work.patient_analysis;
    class meets_tir_goal_num;
    var time_below_70_pct cv_pct n_hypo_events;
    format meets_tir_goal_num tir_fmt.;
run;
title;


/* =========================================================================
   9.  PROC CORR  — correlation matrix of clinical metrics
   ======================================================================= */
title "Table 6. Pearson Correlation Matrix — Clinical Metrics";
proc corr data=work.patient_analysis pearson spearman plots=matrix(nvar=6);
    var time_in_range_pct
        time_below_70_pct
        time_above_180_pct
        cv_pct
        mean_glucose
        n_hypo_events;
run;
title;


/* =========================================================================
   10. PROC REG  — linear regression: TIR predicted by mean glucose and CV
   ======================================================================= */
title "Table 7. Linear Regression — Time-in-Range ~ Mean Glucose + CV";
proc reg data=work.patient_analysis plots=diagnostics;
    model time_in_range_pct = mean_glucose cv_pct / vif clb;
    output out=work.reg_out predicted=pred_tir residual=resid;
    label time_in_range_pct = 'Time in Range %'
          mean_glucose       = 'Mean Glucose (mg/dL)'
          cv_pct             = 'Glucose CV %';
run;
title;


/* =========================================================================
   11. PROC LOGISTIC  — binary outcome: high risk (y=1) vs. low/moderate (y=0)
   ======================================================================= */
data work.logistic_data;
    set work.patient_analysis;
    high_risk = (risk_tier = 'High');
run;

title "Table 8. Logistic Regression — Predictors of High Hypoglycemia Risk";
proc logistic data=work.logistic_data descending plots=roc;
    model high_risk(event='1') = time_below_70_pct cv_pct n_hypo_events / ctable pprob=0.5;
    units time_below_70_pct = 1 cv_pct = 1 n_hypo_events = 5;
    output out=work.log_out predicted=pred_prob;
run;
title;


/* =========================================================================
   12. PROC SGPLOT  — visualizations
   ======================================================================= */

/* 12a. Bar chart: TIR by patient, colour = risk tier */
title "Figure 2. Time-in-Range by Patient";
proc sgplot data=work.patient_analysis;
    vbar subject_id / response=time_in_range_pct
                      group=risk_tier
                      groupdisplay=cluster
                      datalabel;
    refline 70 / axis=y lineattrs=(color=red pattern=dash) label='ADA Goal (70%)';
    yaxis label='Time in Range (%)' min=0 max=100;
    xaxis label='Subject ID';
    keylegend / title='Risk Tier' location=inside position=topright;
run;
title;

/* 12b. Scatter: CV vs TIR — clinical quadrant chart */
title "Figure 3. Glycemic Variability vs. Time-in-Range (Risk Quadrant)";
proc sgplot data=work.patient_analysis;
    scatter x=cv_pct y=time_in_range_pct / group=risk_tier
                                            datalabel=subject_id
                                            markerattrs=(size=10);
    refline 36  / axis=x lineattrs=(color=gray pattern=dash) label='CV=36%';
    refline 70  / axis=y lineattrs=(color=red  pattern=dash) label='TIR=70%';
    xaxis label='Coefficient of Variation (CV %)';
    yaxis label='Time in Range (70-180 mg/dL) %';
    keylegend / title='Risk Tier';
run;
title;

/* 12c. Histogram: hypo events per day */
title "Figure 4. Distribution of Hypoglycemic Events Per Day";
proc sgplot data=work.patient_analysis;
    histogram hypo_events_per_day / fillattrs=(color=steelblue) transparency=0.2;
    density hypo_events_per_day / type=normal lineattrs=(color=red);
    density hypo_events_per_day / type=kernel lineattrs=(color=darkgreen);
    xaxis label='Hypoglycemic Events per Day';
    yaxis label='Frequency';
run;
title;

/* 12d. Box plots: glucose zone time % by risk tier */
title "Figure 5. Glucose Zone Distribution by Risk Tier";
proc sgplot data=work.patient_analysis;
    vbox time_in_range_pct  / category=risk_tier legendlabel='TIR %';
    vbox time_below_70_pct  / category=risk_tier legendlabel='<70 %';
    xaxis label='Risk Tier';
    yaxis label='% of Readings' min=0 max=100;
    format risk_tier $risk_fmt.;
run;
title;


/* =========================================================================
   13. PROC SGPANEL  — faceted glucose timeseries (first 6 subjects)
   ======================================================================= */

/* Prepare timeseries data */
data work.ts_plot;
    set work.glucose_ts;
    where subject_id in (1,2,3,4,5,6);
    format glucose_zone $zone_fmt.;
run;

title "Figure 6. CGM Time Series by Subject (first 6 subjects)";
proc sgpanel data=work.ts_plot noautolegend;
    panelby subject_id / columns=2 rows=3 novarname;
    series x=datetime y=glucose / group=glucose_zone lineattrs=(thickness=1);
    refline 70  / axis=y lineattrs=(color=red    pattern=dash) label='70';
    refline 180 / axis=y lineattrs=(color=orange pattern=dash) label='180';
    colaxis label='Date';
    rowaxis label='Glucose (mg/dL)' min=40 max=350;
run;
title;


/* =========================================================================
   14. Summary macro — output key stats to a results dataset
   ======================================================================= */
%macro cohort_kpis;
    proc sql noprint;
        select count(*)                                     into :n_subjects
            from work.patient_analysis;
        select mean(time_in_range_pct)                      into :mean_tir
            from work.patient_analysis;
        select sum(meets_tir_goal_num) / count(*) * 100     into :pct_tir_goal
            from work.patient_analysis;
        select sum(meets_hypo_goal_num) / count(*) * 100    into :pct_hypo_goal
            from work.patient_analysis;
        select sum(fully_controlled) / count(*) * 100       into :pct_fully_ctrl
            from work.patient_analysis;
    quit;

    %put NOTE: Cohort KPIs —;
    %put NOTE:   N subjects       = &n_subjects;
    %put NOTE:   Mean TIR         = %sysfunc(round(&mean_tir, 0.1))%;
    %put NOTE:   Pct meeting TIR  = %sysfunc(round(&pct_tir_goal, 0.1))%;
    %put NOTE:   Pct safe hypo    = %sysfunc(round(&pct_hypo_goal, 0.1))%;
    %put NOTE:   Fully controlled = %sysfunc(round(&pct_fully_ctrl, 0.1))%;
%mend;

%cohort_kpis;


/* =========================================================================
   15. Close ODS
   ======================================================================= */
ods html close;
ods graphics off;

/*
  OUTPUT FILES:
    sas_output/cgm_clinical_report.html  — full HTML report with tables + figures

  NOTES:
    - Run in SAS Studio (https://odamid.oda.sas.com) for free access to SAS.
    - All data are simulated/de-identified.  Not for clinical decision-making.
    - PROC LOGISTIC / PROC REG results on n=6 (sample data) are illustrative;
      full 25-subject dataset via DVC pull produces meaningful estimates.
*/
