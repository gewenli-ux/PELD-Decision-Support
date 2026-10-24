import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="PELD Decision Support", layout="wide", page_icon="🧠")

# ==========================================
# 1. 全局 CSS 样式 (专为论文截图优化)
# ==========================================
custom_css = """
<style>
html, body, [class*="css"] { font-family: 'Microsoft YaHei', 'Arial', sans-serif !important; font-size: 18px !important; font-weight: 800 !important; color: #000000 !important; }
h1 { font-size: 40px !important; font-weight: 900 !important; color: #000000 !important; margin-bottom: 20px !important; }
h2, h3 { font-size: 28px !important; font-weight: 900 !important; color: #000000 !important; margin-top: 15px !important; }
label { font-size: 20px !important; font-weight: 800 !important; color: #000000 !important; }
.stSelectbox div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input { font-size: 20px !important; font-weight: 800 !important; color: #000000 !important; }
.stButton > button, .stFormSubmitButton > button { font-size: 22px !important; font-weight: 900 !important; padding: 12px 28px !important; color: #FFFFFF !important; background-color: #005A9C !important; }
.stAlert p, .stAlert div { font-size: 20px !important; font-weight: 800 !important; color: #000000 !important; }
[data-testid="stSidebar"] * { font-size: 20px !important; font-weight: 800 !important; }
[data-testid="stMetricValue"] { font-size: 36px !important; font-weight: 900 !important; }
[data-testid="stMetricLabel"] { font-size: 20px !important; font-weight: 800 !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 2. 多语言字典
# ==========================================
TEXTS = {
    "zh": {
        "title": "🧠 PELD 临床决策支持系统 (System 2)",
        "caption": "基于双队列前瞻性研究（n=757）与可解释机器学习模型，输出个体化恢复概率、动态轨迹及预康复处方。",
        "lang": "🌐 Language / 语言",
        "mode_sel": "🎛️ 系统运行模式",
        "mode_pre": "🩺 术前预测模式 (Doctor)",
        "mode_post": "📈 术后追踪模式 (Follow-up)",
        
        "upload_csv": "📂 闭环打通：导入系统1数据",
        "upload_help": "请上传系统1在侧边栏导出的 CSV 宽表数据",
        "select_pid": "👤 请选择要分析的患者编号",
        "no_data": "未导入数据，请手动输入。",
        
        "input_pre": "📋 术前变量输入 (16项)",
        "patient_id": "患者唯一病历编号",
        "sec1": "1. 人口学与临床基线",
        "sec2": "2. 心理、实验室与影像参数",
        
        "age": "年龄 (岁)", "gender": "性别", "bmi": "BMI", "occ": "职业",
        "sym": "症状持续时间 (月)", "neuro": "神经功能缺损分级", "vas": "术前腿部 VAS (0-10)",
        "preop_odi": "术前 ODI (%)", "preop_tug": "术前 TUG (秒)",
        "pcs": "疼痛灾难化量表 (PCS)", "nlr": "中性粒/淋巴比值 (NLR)", "stenosis": "椎管侵占率 (%)",
        "pfirr": "Pfirrmann 分级", "seg_n": "手术节段数量", "seg_l": "手术节段位置", "appr": "手术入路",
        
        "btn_pre": "🚀 生成术前预测报告",
        "analyzing": "正在分析患者",
        
        "post_sec": "术后随访数据录入",
        "select_group": "选择该患者术前的预测组别",
        "grp_zh": ["Rapid 组 (低危/双良好)", "Gradual 组 (中危/渐进)", "Poor 组 (高危/缓慢)"],
        "tp": "当前随访节点",
        "tps": ["术后1周", "术后1个月", "术后3个月", "术后6个月", "术后1年"],
        "act_odi": "当前实测 ODI (%)", "act_tug": "当前实测 TUG (秒)",
        "btn_post": "📊 生成术后比对报告",
        
        "prob": "双良好恢复概率", "risk": "风险分层",
        "risk_low": "低危 (Low Risk)", "risk_med": "中危 (Medium Risk)", "risk_high": "高危 (High Risk)",
        "axis_title": "📊 三轴风险拆解 (基于SHAP阈值)",
        "axis_tug": "躯体功能 (TUG)", "axis_bio": "生物学负担 (NLR+狭窄)", "axis_pcs": "心理困扰 (PCS)",
        "axis_high": "高危", "axis_norm": "正常",
        
        "presc_title": "💊 临床干预处方 (Risk-Tailored)",
        "presc_low": "✅ 推荐常规围手术期照护 (ERAS 路径)。",
        "presc_med": "⚠️ 建议实施针对性预康复：",
        "presc_high": "🚨 提示存在多重预后障碍，强烈建议启动多学科评估 (MDT) 与强化术前优化。",
        "presc_pcs": "👉 心理干预：患者存在高 PCS，建议术前实施认知行为疗法 (CBT)。",
        "presc_tug": "👉 物理康复：TUG 延长，建议加入术前物理康复及抗炎预康复方案。",
        
        "traj_pre": "📈 术前预期功能恢复轨迹",
        "traj_post": "📈 动态恢复轨迹与预期比对",
        "fig_odi": "预期 ODI (主观)", "fig_tug": "预期 TUG (客观)",
        "fig_act_odi": "实测 ODI", "fig_act_tug": "实测 TUG",
        "fig_x": "随访时间点", "fig_y": "评分/时间",
        "fig_rebound": "⚠️ 6个月症状反弹高发期",
        "time_arr": ["术前", "术后1周", "术后1月", "术后3月", "术后6月", "术后1年"],
        
        "concl_pre": "该患者恢复轨迹预测属于", "concl_grp": "组",
        "concl_post": "术后随访偏离度分析",
        "concl_exp": "预期值", "concl_act": "实测",
        
        "alert_dev_both": "🚨 严重偏离预警：主观 ODI 与客观 TUG 均显著差于术前预期轨迹！建议立即进行全面临床评估。",
        "alert_dev_tug": "🚨 偏离预警：实测 TUG 显著慢于预期。提示客观行动能力未达标，建议复查并加强物理康复。",
        "alert_sep_1": "⚠️ 主客观分离警报：患者自报症状（ODI）较差，但客观行走能力尚可。需警惕心理因素（如高PCS）导致过度担忧。",
        "alert_sep_2": "⚠️ 主客观分离警报：患者自感良好，但客观行走功能显著落后。建议复查MRI排查神经压迫，并加强物理康复。",
        "alert_ok": "✅ 患者当前恢复轨迹平稳，与术前预期高度相符，建议按常规计划随访。",
        "alert_better_both": "🌟 极佳恢复：实测 ODI 与 TUG 均显著优于术前预期轨迹！患者恢复状况极好。",
        "alert_better_one": "✅ 恢复良好：患者当前实测数据达到或优于术前预期轨迹，建议按常规计划随访。",
        
        "info_wait_pre": "👈 请在左侧输入患者信息或导入CSV，点击【生成术前预测报告】。",
        "info_wait_post": "👈 请选择术前组别与随访节点，输入实测数据，点击【生成术后比对报告】。"
    },
    "en": {
        "title": "🧠 PELD Clinical Decision Support (System 2)",
        "caption": "Based on prospective dual-cohort study (n=757) and interpretable ML, outputting individualized probabilities, trajectories, and prescriptions.",
        "lang": "🌐 Language / 语言",
        "mode_sel": "🎛️ System Mode",
        "mode_pre": "🩺 Preoperative Prediction",
        "mode_post": "📈 Postoperative Tracking",
        
        "upload_csv": "📂 Closed-Loop: Import System 1 Data",
        "upload_help": "Upload the wide-format CSV exported from System 1",
        "select_pid": "👤 Select Patient ID to analyze",
        "no_data": "No data imported. Manual entry.",
        
        "input_pre": "📋 Preoperative Variables (16)",
        "patient_id": "Patient ID",
        "sec1": "1. Demographic & Clinical",
        "sec2": "2. Psych, Lab & Imaging Params",
        
        "age": "Age (years)", "gender": "Gender", "bmi": "BMI", "occ": "Occupation",
        "sym": "Symptom Duration (months)", "neuro": "Neurological Deficit Grade", "vas": "Preop VAS_Leg (0-10)",
        "preop_odi": "Preop ODI (%)", "preop_tug": "Preop TUG (sec)",
        "pcs": "Pain Catastrophizing Scale (PCS)", "nlr": "NLR", "stenosis": "Canal Stenosis (%)",
        "pfirr": "Pfirrmann Grade", "seg_n": "Number of surgical levels", "seg_l": "Segs. Location", "appr": "Surgical Approach",
        
        "btn_pre": "🚀 Generate Preop Report",
        "analyzing": "Analyzing Patient",
        
        "post_sec": "Postoperative Data Entry",
        "select_group": "Select Preoperative Predicted Group",
        "grp_zh": ["Rapid Group (Low Risk)", "Gradual Group (Medium Risk)", "Poor Group (High Risk)"],
        "tp": "Current Follow-up",
        "tps": ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
        "act_odi": "Actual ODI (%)", "act_tug": "Actual TUG (sec)",
        "btn_post": "📊 Generate Tracking Report",
        
        "prob": "Dual Good Probability", "risk": "Risk Tier",
        "risk_low": "Low Risk", "risk_med": "Medium Risk", "risk_high": "High Risk",
        "axis_title": "📊 Three-Axis Decomposition (SHAP)",
        "axis_tug": "Physical (TUG)", "axis_bio": "Bio-burden (NLR+Stenosis)", "axis_pcs": "Psych (PCS)",
        "axis_high": "High", "axis_norm": "Normal",
        
        "presc_title": "💊 Clinical Prescription",
        "presc_low": "✅ Routine perioperative care (ERAS).",
        "presc_med": "⚠️ Targeted prehabilitation recommended:",
        "presc_high": "🚨 Intensive Optimization: Consider postponing non-emergency surgery and starting Multidisciplinary Team (MDT).",
        "presc_pcs": "👉 Psych support: CBT recommended due to high PCS.",
        "presc_tug": "👉 Physical rehab: Consider physical therapy and anti-inflammatory prehab for prolonged TUG.",
        
        "traj_pre": "📈 Preoperative Expected Trajectory",
        "traj_post": "📈 Dynamic Trajectory Comparison",
        "fig_odi": "Expected ODI", "fig_tug": "Expected TUG",
        "fig_act_odi": "Actual ODI", "fig_act_tug": "Actual TUG",
        "fig_x": "Timepoint", "fig_y": "Score/Time",
        "fig_rebound": "⚠️ 6-Month Rebound Risk",
        "time_arr": ["Preop", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
        
        "concl_pre": "Predicted trajectory group:", "concl_grp": "",
        "concl_post": "Trajectory Deviation Analysis",
        "concl_exp": "Expected", "concl_act": "Actual",
        
        "alert_dev_both": "🚨 Severe Deviation: Both ODI and TUG are significantly worse than expected! Immediate clinical evaluation required.",
        "alert_dev_tug": "🚨 Deviation Alert: Actual TUG is significantly slower than expected. Consider physical rehab.",
        "alert_sep_1": "⚠️ Discordance Alert: Poor subjective symptoms (ODI), but objective function meets target. Watch for psychological over-worry.",
        "alert_sep_2": "⚠️ Discordance Alert: Good subjective feeling, but objective function lags. Consider MRI and physical rehab.",
        "alert_ok": "✅ Patient on stable trajectory and aligns with preoperative expectation. Continue routine follow-up.",
        "alert_better_both": "🌟 Excellent Recovery: Both ODI and TUG are significantly better than expected!",
        "alert_better_one": "✅ Good Recovery: Actual data meets or exceeds expected trajectory. Continue routine follow-up.",
        
        "info_wait_pre": "👈 Input patient info or import CSV on the left, then click Generate.",
        "info_wait_post": "👈 Select group and timepoint, input actual data, then generate."
    }
}

OPTS = {
    "zh": {
        "gender": ["男", "女"], "occ": ["久坐", "站立", "轻体力", "退休", "重体力"],
        "neuro": ["0 - 无", "1 - 轻度", "2 - 中度", "3 - 重度"], "pfirr": ["1级", "2级", "3级", "4级", "5级"],
        "seg_n": ["单节段", "双节段", "≥3节段"], "seg_l": ["L3/4", "L4/5", "L5/S1", "混合"],
        "appr": ["椎间孔入路", "椎板间入路"]
    },
    "en": {
        "gender": ["Male", "Female"], "occ": ["Sedentary", "Standing", "Light manual", "Retired", "Heavy manual"],
        "neuro": ["0 - None", "1 - Mild", "2 - Moderate", "3 - Severe"], "pfirr": ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"],
        "seg_n": ["Single", "Double", "≥3 Segments"], "seg_l": ["L3/4", "L4/5", "L5/S1", "Mixed"],
        "appr": ["Transforaminal", "Interlaminar"]
    }
}

# ==========================================
# 3. 真实轨迹数据
# ==========================================
TIME_MAP = {"术后1周": 1, "1 Week": 1, "术后1个月": 2, "1 Month": 2, "术后3个月": 3, "3 Months": 3, "术后6个月": 4, "6 Months": 4, "术后1年": 5, "1 Year": 5}

TRAJ = {
    "Rapid": {"ODI": [45.17, 20.45, 16.92, 16.41, 31.39, 23.49], "TUG": [15.34, 14.59, 13.07, 11.33, 10.12, 11.46]},
    "Gradual": {"ODI": [47.90, 31.90, 20.60, 23.00, 36.73, 26.39], "TUG": [18.91, 17.97, 16.43, 14.54, 13.12, 14.40]},
    "Poor": {"ODI": [60.79, 29.33, 25.56, 21.77, 35.11, 28.09], "TUG": [16.46, 14.96, 14.24, 12.61, 11.42, 12.72]}
}

# ==========================================
# 4. 侧边栏与数据输入
# ==========================================
def_pid = "P-1001"
def_age, def_bmi, def_odi, def_tug = 52, 26.0, 52.0, 16.4
def_pcs, def_nlr, def_steno, def_vas, def_sym = 20, 1.8, 40.0, 7.5, 21

with st.sidebar:
    lang_choice = st.selectbox(TEXTS["zh"]["lang"], ["中文", "English"])
    lang = "zh" if lang_choice == "中文" else "en"
    t = TEXTS[lang]
    o = OPTS[lang]
    
    def_gender, def_occ, def_neuro, def_pfirr = o["gender"][0], o["occ"][0], o["neuro"][0], o["pfirr"][2]
    def_seg_n, def_seg_l, def_appr = o["seg_n"][0], o["seg_l"][1], o["appr"][0]
    
    # 数据导入
    st.markdown(f"### {t['upload_csv']}")
    uploaded_file = st.file_uploader(t['upload_help'], type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'PatientID' in df.columns:
                pid_list = df['PatientID'].astype(str).tolist()
                selected_pid = st.selectbox(t['select_pid'], pid_list)
                row = df[df['PatientID'].astype(str) == selected_pid].iloc[0]
                def_pid = selected_pid
            else:
                row = df.iloc[0]
            
            def_age = int(row.get("Age", 52))
            def_bmi = float(row.get("BMI", 26.0))
            def_odi = float(row.get("Preop_ODI", 52.0))
            def_tug = float(row.get("Preop_TUG", 16.4))
            def_pcs = int(row.get("Pain Catastrophizing Scale (PCS)", 20))
            def_nlr = float(row.get("NLR", 1.8))
            def_steno = float(row.get("Canal_Compromise", 40.0))
            def_vas = float(row.get("Preop_VAS_Leg", 7.5))
            def_sym = int(row.get("Symptom_Duration", 21))
            
            g_val = str(row.get("Gender", o["gender"][0]))
            def_gender = g_val if g_val in o["gender"] else o["gender"][0]
            occ_val = str(row.get("Occupation", o["occ"][0]))
            def_occ = occ_val if occ_val in o["occ"] else o["occ"][0]
            neuro_val = str(row.get("Neurological_Deficit", o["neuro"][0]))
            def_neuro = neuro_val if neuro_val in o["neuro"] else o["neuro"][0]
            pf_val = str(row.get("Pfirrmann_Grade", o["pfirr"][2]))
            def_pfirr = pf_val if pf_val in o["pfirr"] else o["pfirr"][2]
            seg_n_val = str(row.get("Number of surgical levels", o["seg_n"][0]))
            def_seg_n = seg_n_val if seg_n_val in o["seg_n"] else o["seg_n"][0]
            seg_l_val = str(row.get("Segs. Location", o["seg_l"][1]))
            def_seg_l = seg_l_val if seg_l_val in o["seg_l"] else o["seg_l"][1]
            appr_val = str(row.get("surgical approach", o["appr"][0]))
            def_appr = appr_val if appr_val in o["appr"] else o["appr"][0]
            
            st.success(f"✅ 数据导入成功！")
        except Exception as e:
            st.error(f"读取失败: {e}")
    else:
        st.info(t['no_data'])
    
    st.markdown("---")
    mode = st.radio(t["mode_sel"], [t["mode_pre"], t["mode_post"]], label_visibility="collapsed")
    st.markdown("---")
    
    if mode == t["mode_pre"]:
        st.header(t["input_pre"])
        
        pid_input = st.text_input(t["patient_id"], value=def_pid)
        
        st.subheader(t["sec1"])
        age = st.number_input(t["age"], 20, 90, def_age)
        gender = st.selectbox(t["gender"], o["gender"], index=o["gender"].index(def_gender))
        bmi = st.number_input(t["bmi"], 15.0, 40.0, def_bmi)
        occ = st.selectbox(t["occ"], o["occ"], index=o["occ"].index(def_occ))
        sym = st.number_input(t["sym"], 0, 200, def_sym)
        neuro = st.selectbox(t["neuro"], o["neuro"], index=o["neuro"].index(def_neuro))
        vas = st.number_input(t["vas"], 0.0, 10.0, def_vas)
        preop_odi = st.number_input(t["preop_odi"], 0.0, 100.0, def_odi)
        preop_tug = st.number_input(t["preop_tug"], 5.0, 30.0, def_tug)
        
        st.subheader(t["sec2"])
        pcs = st.number_input(t["pcs"], 0, 52, def_pcs)
        nlr = st.number_input(t["nlr"], 0.0, 20.0, def_nlr)
        stenosis = st.number_input(t["stenosis"], 0.0, 100.0, def_steno)
        pfirr = st.selectbox(t["pfirr"], o["pfirr"], index=o["pfirr"].index(def_pfirr))
        seg_n = st.selectbox(t["seg_n"], o["seg_n"], index=o["seg_n"].index(def_seg_n))
        seg_l = st.selectbox(t["seg_l"], o["seg_l"], index=o["seg_l"].index(def_seg_l))
        appr = st.selectbox(t["appr"], o["appr"], index=o["appr"].index(def_appr))
        
        st.markdown("---")
        predict_btn = st.button(t["btn_pre"], use_container_width=True)
        
    else:
        st.header(t["post_sec"])
        pid_input = st.text_input(t["patient_id"], value=def_pid)
        grp_list = t["grp_zh"] if lang == "zh" else t["grp_en"]
        selected_grp_str = st.selectbox(t["select_group"], grp_list)
        traj_group = "Rapid" if "Rapid" in selected_grp_str else ("Gradual" if "Gradual" in selected_grp_str else "Poor")
        st.markdown("---")
        current_tp_name = st.selectbox(t["tp"], t["tps"])
        actual_odi = st.number_input(t["act_odi"], 0.0, 100.0, 20.0)
        actual_tug = st.number_input(t["act_tug"], 5.0, 30.0, 15.0)
        st.markdown("---")
        track_btn = st.button(t["btn_post"], use_container_width=True)

st.title(t["title"])
st.caption(t["caption"])

# ==========================================
# 5. 核心业务逻辑渲染
# ==========================================
if mode == t["mode_pre"] and predict_btn:
    st.markdown(f"#### 👤 {t['analyzing']}: **{pid_input}**")
    
    # 严格套用正文的 0-6 分三轴矩阵逻辑
    s_tug = 2 if preop_tug > 17 else (1 if preop_tug > 14 else 0)
    
    bio_score = 0
    if nlr > 2.8: bio_score += 1
    if stenosis > 50: bio_score += 1
    s_bio = 2 if bio_score == 2 else (1 if bio_score == 1 else 0)
    
    s_pcs = 2 if pcs > 30 else (1 if pcs > 20 else 0)
    
    total_score = s_tug + s_bio + s_pcs
    
    # 修正规则：躯体功能0分(快速)且心理困扰2分(高PCS)时，总分减1分
    if s_tug == 0 and s_pcs == 2:
        total_score -= 1
        
    # 映射概率和轨迹
    if total_score <= 1:
        risk, prob_str, color, traj_group = t["risk_low"], "49.5%", "green", "Rapid"
    elif total_score <= 3:
        risk, prob_str, color, traj_group = t["risk_med"], "25.4%", "orange", "Gradual"
    else:
        risk, prob_str, color, traj_group = t["risk_high"], "3.5%", "red", "Poor"

    st.markdown(f"### {t['prob']}：<span style='color:{color}; font-size:38px;'>{prob_str}</span>", unsafe_allow_html=True)
    st.markdown(f"### {t['risk']}：<span style='color:{color}; font-size:34px;'>{risk}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(t["axis_title"])
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(t["axis_tug"], f"{s_tug}", t["axis_high"] if s_tug==2 else t["axis_norm"], delta_color="inverse")
    col_b.metric(t["axis_bio"], f"{s_bio}", t["axis_high"] if s_bio==2 else t["axis_norm"], delta_color="inverse")
    col_c.metric(t["axis_pcs"], f"{s_pcs}", t["axis_high"] if s_pcs==2 else t["axis_norm"], delta_color="inverse")

    st.markdown("---")
    st.subheader(t["presc_title"])
    if risk == t["risk_low"]:
        st.success(t["presc_low"])
    else:
        if risk == t["risk_high"]:
            st.error(t["presc_high"])
        else:
            st.warning(t["presc_med"])
        if s_pcs >= 1:
            st.info(t["presc_pcs"])
        if s_tug >= 1:
            st.info(t["presc_tug"])

    st.markdown("---")
    st.subheader(t["traj_pre"])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["time_arr"], y=TRAJ[traj_group]['ODI'], name=t["fig_odi"], line=dict(color='red', width=4)))
    fig.add_trace(go.Scatter(x=t["time_arr"], y=TRAJ[traj_group]['TUG'], name=t["fig_tug"], line=dict(color='blue', width=4)))
    
    fig.add_vline(x=4, line_width=2, line_dash="dash", line_color="gray")
    fig.add_annotation(
        x=4, y=max(TRAJ[traj_group]['ODI']) + 8,
        text=t["fig_rebound"], showarrow=False, 
        font=dict(size=16, color="red", weight="bold"),
        bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="red", borderwidth=1
    )
    fig.update_layout(
        xaxis_title=t["fig_x"], yaxis_title=t["fig_y"], 
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1), 
        margin=dict(l=0, r=0, t=50, b=0), font=dict(size=14, weight='bold')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**{t['concl_pre']} {traj_group} {t['concl_grp']}**")

elif mode == t["mode_post"] and track_btn:
    st.markdown(f"#### 👤 {t['analyzing']}: **{pid_input}**")
    st.markdown(f"### {t['concl_post']}")
    
    idx = TIME_MAP[current_tp_name]
    expected_odi = TRAJ[traj_group]['ODI'][idx]
    expected_tug = TRAJ[traj_group]['TUG'][idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["time_arr"], y=TRAJ[traj_group]['ODI'], name=t["fig_odi"], line=dict(color='red', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=t["time_arr"], y=TRAJ[traj_group]['TUG'], name=t["fig_tug"], line=dict(color='blue', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=[current_tp_name], y=[actual_odi], mode='markers', name=t["fig_act_odi"], marker=dict(color='red', size=20, symbol='star')))
    fig.add_trace(go.Scatter(x=[current_tp_name], y=[actual_tug], mode='markers', name=t["fig_act_tug"], marker=dict(color='blue', size=20, symbol='star')))
    fig.add_vline(x=4, line_width=1, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        xaxis_title=t["fig_x"], yaxis_title=t["fig_y"], 
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1), 
        margin=dict(l=0, r=0, t=50, b=0), font=dict(size=14, weight='bold')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    
    odi_dev = actual_odi - expected_odi
    tug_dev = actual_tug - expected_tug
    worse_odi = odi_dev > 8      
    worse_tug = tug_dev > 2.5    
    better_odi = odi_dev < -5    
    better_tug = tug_dev < -1.5  

    odi_str = f"- **{t['fig_act_odi']}**：{t['concl_exp']} {expected_odi:.1f}% , {t['concl_act']} {actual_odi:.1f}%。"
    tug_str = f"- **{t['fig_act_tug']}**：{t['concl_exp']} {expected_tug:.1f}s , {t['concl_act']} {actual_tug:.1f}s。"
    st.write(odi_str)
    st.write(tug_str)
    
    if worse_odi and worse_tug:
        st.error(t["alert_dev_both"])
    elif worse_odi and not worse_tug:
        st.warning(t["alert_sep_1"])
    elif worse_tug and not worse_odi:
        st.error(t["alert_dev_tug"])
    elif better_odi and better_tug:
        st.success(t["alert_better_both"])
    elif better_odi or better_tug:
        st.success(t["alert_better_one"])
    else:
        st.success(t["alert_ok"])

else:
    if mode == t["mode_pre"]:
        st.info(t["info_wait_pre"])
    else:
        st.info(t["info_wait_post"])
