import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap


# Configure page
st.set_page_config(
    page_title="Durham County Junior Cricket Player Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        margin: 1rem 0;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #b8daff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

#Title
st.markdown('<h1 class="main-header">Durham County Junior Cricket Player Analysis Dashboard</h1>', unsafe_allow_html=True)

#Sidebar 
st.sidebar.title("Navigation")
analysis_type = st.sidebar.selectbox(
    "Select Analysis",
    ["Overview", "PCA Analysis", "K-Means Clustering", "XGBoost & SHAP", "Player Comparison", "Insights & Recommendations"]
)

#Load data function
@st.cache_data
def load_data(uploaded_file=None):

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None
    else:
        try:
            # Try to load from file system first
            data = pd.read_csv('final_data.csv')
            return data
        except FileNotFoundError:
            return None
        except Exception as e:
            st.error(f"Error reading final_data.csv: {e}")
            return None

#File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv", help="Upload your final_data.csv file")

#Load data
data = load_data(uploaded_file)

if data is not None:
    # Data preprocessing
    @st.cache_data
    def preprocess_data(df):
        # Separate features and target
        features = df.select_dtypes(include=[np.number]).drop(['SELECTION'], axis=1, errors='ignore')
        features = features.fillna(0)
        
        # Remove zero variance columns
        features = features.loc[:, features.var() != 0]
        
        return features, df['SELECTION'] if 'SELECTION' in df.columns else None
    
    features, selection = preprocess_data(data)
    
    #Overview Tab
    if analysis_type == "Overview":
        st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        #Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Players", len(data))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if selection is not None:
                selected_count = selection.sum()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Selected Players", selected_count)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Features Analyzed", len(features.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            if selection is not None:
                selection_rate = (selection.sum() / len(selection) * 100)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Selection Rate", f"{selection_rate:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        #Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        #Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Performance Statistics")
            key_stats = ['GAMES', 'RUNS', 'AVG', 'STRIKE RATE', 'WICKETS', 'ECONOMY RATE']
            available_stats = [stat for stat in key_stats if stat in data.columns]
            if available_stats:
                st.dataframe(data[available_stats].describe(), use_container_width=True)
        
        with col2:
            if selection is not None:
                st.subheader("Selection Distribution")
                selection_dist = pd.DataFrame({
                    'Status': ['Selected', 'Not Selected'],
                    'Count': [selection.sum(), len(selection) - selection.sum()]
                })
                fig = px.pie(selection_dist, values='Count', names='Status', 
                           color_discrete_sequence=['#FDE725', '#440154'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    #PCA Analysis Tab
    elif analysis_type == "PCA Analysis":
        st.markdown('<h2 class="sub-header">Principal Component Analysis</h2>', unsafe_allow_html=True)
        
        #Perform PCA
        @st.cache_data
        def perform_pca(features):
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            pca = PCA()
            pca_result = pca.fit_transform(features_scaled)
            
            return pca, pca_result, scaler
        
        pca, pca_result, scaler = perform_pca(features)
        
        #PCA Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            #Variance explained
            variance_explained = pca.explained_variance_ratio_ * 100
            cumulative_variance = np.cumsum(variance_explained)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, min(11, len(variance_explained) + 1))),
                y=variance_explained[:10],
                name='Individual Variance',
                marker_color='#1f4e79'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, min(11, len(variance_explained) + 1))),
                y=cumulative_variance[:10],
                mode='lines+markers',
                name='Cumulative Variance',
                yaxis='y2',
                marker_color='#FDE725'
            ))
            
            fig.update_layout(
                title="PCA Variance Explained",
                xaxis_title="Principal Component",
                yaxis_title="Variance Explained (%)",
                yaxis2=dict(title="Cumulative Variance (%)", overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            #PCA scatter plot
            if selection is not None:
                pca_df = pd.DataFrame({
                    'PC1': pca_result[:, 0],
                    'PC2': pca_result[:, 1],
                    'Selection': ['Selected' if s == 1 else 'Not Selected' for s in selection],
                    'Player': data['Player'] if 'Player' in data.columns else range(len(data))
                })
                
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Selection',
                               title=f"PCA Plot (PC1: {variance_explained[0]:.1f}%, PC2: {variance_explained[1]:.1f}%)",
                               color_discrete_map={'Selected': '#FDE725', 'Not Selected': '#440154'},
                               hover_data=['Player'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        #Component loadings
        st.subheader("Principal Component Loadings")
        
        n_components = min(3, pca.n_components_)
        loadings_df = pd.DataFrame(
            pca.components_[:n_components].T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=features.columns
        )
        
        #Show top contributing variables
        for i in range(n_components):
            pc_name = f'PC{i+1}'
            top_loadings = loadings_df[pc_name].abs().nlargest(10)
            
            col1, col2 = st.columns(2)
            with col1 if i % 2 == 0 else col2:
                st.write(f"**Top 10 Contributors to {pc_name}**")
                loadings_display = pd.DataFrame({
                    'Variable': top_loadings.index,
                    'Loading': loadings_df.loc[top_loadings.index, pc_name].round(3)
                })
                st.dataframe(loadings_display, use_container_width=True)
    
    #K-Means Clustering Tab
    elif analysis_type == "K-Means Clustering":
        st.markdown('<h2 class="sub-header">K-Means Clustering Analysis</h2>', unsafe_allow_html=True)
        
        #Clustering analysis
        @st.cache_data
        def perform_clustering(features, max_k=10):
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            #Elbow method
            inertias = []
            silhouette_scores = []
            k_range = range(2, max_k + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features_scaled)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
            
            return features_scaled, inertias, silhouette_scores, k_range
        
        features_scaled, inertias, silhouette_scores, k_range = perform_clustering(features)
        
        #Optimal k selection
        col1, col2 = st.columns(2)
        
        with col1:
            #Elbow curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                                   marker=dict(color='#1f4e79'), line=dict(width=3)))
            fig.update_layout(title="Elbow Method", xaxis_title="Number of Clusters (k)",
                            yaxis_title="Inertia", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            #Silhouette scores
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                                   marker=dict(color='#FDE725'), line=dict(width=3)))
            fig.update_layout(title="Silhouette Analysis", xaxis_title="Number of Clusters (k)",
                            yaxis_title="Silhouette Score", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        #Optimal k selection widget
        optimal_k_suggestion = k_range[np.argmax(silhouette_scores)]
        k_selected = st.selectbox(
            f"Select number of clusters (Suggested: {optimal_k_suggestion})",
            options=list(k_range),
            index=list(k_range).index(optimal_k_suggestion)
        )
        
        #Perform clustering with selected k
        kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        #Add clusters to dataframe
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = cluster_labels
        
        #PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        
        #Cluster visualization
        cluster_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': [f'Cluster {c}' for c in cluster_labels],
            'Selection': ['Selected' if s == 1 else 'Not Selected' for s in selection] if selection is not None else ['Unknown'] * len(cluster_labels),
            'Player': data['Player'] if 'Player' in data.columns else range(len(data))
        })
        
        fig = px.scatter(cluster_df, x='PC1', y='PC2', color='Cluster', symbol='Selection',
                        title="K-Means Clusters in PCA Space",
                        hover_data=['Player'])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        #Cluster analysis
        st.subheader("Cluster Analysis")
        
        if selection is not None:
            cluster_analysis = data_with_clusters.groupby('Cluster').agg({
                'SELECTION': ['count', 'sum', lambda x: f"{(x.sum()/len(x)*100):.1f}%"],
                'GAMES': 'mean',
                'RUNS': 'mean',
                'AVG': 'mean',
                'WICKETS': 'mean'
            }).round(2)
            
            cluster_analysis.columns = ['Total Players', 'Selected', 'Selection Rate', 
                                      'Avg Games', 'Avg Runs', 'Avg Batting Avg', 'Avg Wickets']
            st.dataframe(cluster_analysis, use_container_width=True)
            
            #Representative players for each cluster
            st.subheader("Representative Players by Cluster")
            for cluster_id in range(k_selected):
                cluster_players = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
                if len(cluster_players) > 0:
                    st.write(f"**Cluster {cluster_id}** ({len(cluster_players)} players):")
                    sample_players = cluster_players.head(5)['Player'].tolist() if 'Player' in cluster_players.columns else [f"Player {i}" for i in range(min(5, len(cluster_players)))]
                    st.write(", ".join(sample_players))
    
    #XGBoost & SHAP Tab
    elif analysis_type == "XGBoost & SHAP":
        st.markdown('<h2 class="sub-header">XGBoost & SHAP Analysis</h2>', unsafe_allow_html=True)
        
        
        #Check if SELECTION column exists (dataset must be in a compatible format)
        if 'SELECTION' not in data.columns:
            st.error("SELECTION column not found in the dataset")
            st.info("Please ensure your dataset has a 'SELECTION' column with 1 for selected and 0 for not selected players")
        else:
            #Button to run analysis
            if st.button("Run XGBoost Ensemble Analysis", type="primary"):
                models = []    
                with st.spinner("Training ensemble models..."):
                    # Prepare the data
                    X = data[features.columns]
                    y = data["SELECTION"]
                    
                    #Train/Test Split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                    
                    #Ensemble Models Setup
                    n = 5
                    ptrain = np.zeros(X_train.shape[0])
                    ptest = np.zeros(X_test.shape[0])
                    pfull = np.zeros(X.shape[0])
                    
                    
                    progress_bar = st.progress(0)
                    
                    for s in range(n):
                        model = xgb.XGBClassifier(
                            eval_metric="logloss",
                            max_depth=4,
                            learning_rate=0.05,
                            n_estimators=150,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=s
                        )
                        model.fit(X_train, y_train)
                        models.append(model)
                        
                        #Accumulating all predictions
                        ptrain = ptrain + model.predict_proba(X_train)[:, 1]
                        ptest = ptest + model.predict_proba(X_test)[:, 1]
                        pfull = pfull + model.predict_proba(X)[:, 1]
                        
                        progress_bar.progress((s + 1) / n)
                    
                    #Average Predictions
                    ptrain = ptrain/n
                    ptest = ptest/n
                    pfull = pfull/n
                    
                    #Evaluating on the Test Set
                    y_pred = (ptest >= 0.5).astype(int)
                    test_accuracy = accuracy_score(y_test, y_pred)
                    
                    #Storing results in session state
                    st.session_state.models = models
                    st.session_state.X = X
                    st.session_state.pfull = pfull
                    st.session_state.test_accuracy = test_accuracy
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    
                    st.success(f"Ensemble Training Completed! Test Accuracy: {test_accuracy:.3f}")
            
            #Display results if models are trained
            if 'models' in st.session_state:
                models = st.session_state.models
                X = st.session_state.X
                pfull = st.session_state.pfull
                test_accuracy = st.session_state.test_accuracy
                y_test = st.session_state.y_test
                y_pred = st.session_state.y_pred
                
                # Model Performance Section
                st.markdown("---")
                st.subheader("Model Performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Accuracy", f"{test_accuracy:.3f}")
                with col2:
                    precision = (y_pred[y_test == 1] == 1).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
                    st.metric("Precision", f"{precision:.3f}")
                with col3:
                    recall = (y_pred[y_test == 1] == 1).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0
                    st.metric("Recall", f"{recall:.3f}")
                
                #Prediction Distributions
                col1, col2 = st.columns(2)
                
                with col1:
                    #Prediction probability distribution
                    pred_df = pd.DataFrame({
                        'Prediction_Probability': pfull,
                        'Actual_Selection': ['Selected' if s == 1 else 'Not Selected' for s in data['SELECTION']]
                    })
                    
                    fig = px.histogram(pred_df, x='Prediction_Probability', color='Actual_Selection',
                                     title="Prediction Probability Distribution", nbins=20, barmode='overlay',
                                     color_discrete_map={'Selected': '#FDE725', 'Not Selected': '#440154'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    #Feature Importance from Ensemble
                    importances = np.zeros(X.shape[1])
                    for model in models:
                        importances += model.feature_importances_
                    importances /= len(models)
                    
                    #Top 15 features
                    feature_imp_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True).tail(15)
                    
                    fig = go.Figure(go.Bar(
                        x=feature_imp_df['Importance'],
                        y=feature_imp_df['Feature'],
                        orientation='h',
                        marker_color='#1f4e79'
                    ))
                    fig.update_layout(title="Ensemble Feature Importance", 
                                    xaxis_title="Average Importance",
                                    height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                #Top 20 Players Section
                st.markdown("---")
                st.subheader("Top 20 Predicted Players")
                
                #Create results dataframe
                results_df = data.copy()
                results_df["Prediction_Prob"] = pfull
                top20 = results_df.sort_values(by="Prediction_Prob", ascending=False).head(20)
                
                # Display top 20
                display_cols = ['Player', 'Prediction_Prob', 'SELECTION', 'GAMES', 'RUNS', 'AVG', 'WICKETS']
                available_cols = [col for col in display_cols if col in top20.columns]
                
                top20_display = top20[available_cols].copy()
                top20_display['Prediction_Prob'] = top20_display['Prediction_Prob'].round(3)
                top20_display['Rank'] = range(1, 21)
                
                #Reordering columns to put Rank first
                cols = top20_display.columns.tolist()
                cols = ['Rank'] + [col for col in cols if col != 'Rank']
                top20_display = top20_display[cols]
                
                st.dataframe(top20_display, use_container_width=True)
            
            #SHAP Analysis Section
            st.markdown("---")
            st.subheader("SHAP Analysis")
            
            # Check if models exist in session state before allowing SHAP analysis
            if 'models' in st.session_state:
                if st.button("Generate SHAP Analysis"):
                    print("Generating")
                    with st.spinner("Generating SHAP values..."):
                        # Get models and data from session state
                        models = st.session_state.models
                        X = st.session_state.X
                        
                        #Use the first model for SHAP analysis
                        explainer = shap.TreeExplainer(models[0])
                        shap_values = explainer.shap_values(X.iloc[:100])  # Limit to first 100 for performance
                        
                        #SHAP Summary Plot Data
                        shap_summary = pd.DataFrame({
                            'Feature': X.columns,
                            'Mean_SHAP': np.abs(shap_values).mean(axis=0)
                        }).sort_values('Mean_SHAP', ascending=True).tail(15)
                        
                        #SHAP Feature Importance
                        fig = go.Figure(go.Bar(
                            x=shap_summary['Mean_SHAP'],
                            y=shap_summary['Feature'],
                            orientation='h',
                            marker_color='#440154'
                        ))
                        fig.update_layout(title="SHAP Feature Importance", 
                                        xaxis_title="Mean |SHAP Value|",
                                        height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        #Individual player SHAP explanation
                        st.subheader("Individual Player Analysis")
                        player_idx = st.selectbox("Select player index for SHAP explanation:", 
                                                    range(min(20, len(X))))
                        
                        if player_idx is not None:
                            #Create SHAP waterfall-like explanation
                            player_shap = shap_values[player_idx]
                            player_features = X.iloc[player_idx]
                            
                            shap_explanation = pd.DataFrame({
                                'Feature': X.columns,
                                'Value': player_features.values,
                                'SHAP_Value': player_shap
                            }).sort_values('SHAP_Value', key=abs, ascending=False).head(10)
                            
                            fig = go.Figure(go.Bar(
                                x=shap_explanation['SHAP_Value'],
                                y=shap_explanation['Feature'],
                                orientation='h',
                                marker_color=['#FDE725' if x > 0 else '#440154' for x in shap_explanation['SHAP_Value']]
                            ))
                            fig.update_layout(title=f"SHAP Values for Player {player_idx}", 
                                            xaxis_title="SHAP Value",
                                            height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            #Display player details
                            st.write("**Player Details:**")
                            player_details = shap_explanation[['Feature', 'Value', 'SHAP_Value']].round(3)
                            st.dataframe(player_details, use_container_width=True)
            else:
                st.info("⚠️ Please run the XGBoost Ensemble Analysis first before generating SHAP analysis!")
                st.markdown("""
                <div class="insight-box">
                <h4>SHAP Analysis Requirements:</h4>
                <ul>
                <li>First click <strong>"Run XGBoost Ensemble Analysis"</strong> button above</li>
                <li>Wait for the models to train completely</li>
                <li>Then return here to generate SHAP explanations</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            if 'models' not in st.session_state:
                st.info("Click the button above to train the XGBoost ensemble and generate predictions!")
                st.markdown("""
                <div class="insight-box">
                <h4>What this analysis will do:</h4>
                <ul>
                <li><strong>Train 5 XGBoost models</strong> with different random seeds for ensemble prediction</li>
                <li><strong>Generate probability scores</strong> for each player's likelihood of selection</li>
                <li><strong>Identify top 20 players</strong> most likely to be selected based on performance</li>
                <li><strong>SHAP analysis</strong> to explain which features contribute most to predictions</li>
                <li><strong>Individual player insights</strong> showing what drives each player's score</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    #Player Comparison Tab
    elif analysis_type == "Player Comparison":
        st.markdown('<h2 class="sub-header">Player Comparison Tool</h2>', unsafe_allow_html=True)
        
        if 'Player' in data.columns:
            #Player selection
            col1, col2 = st.columns(2)
            
            with col1:
                player1 = st.selectbox("Select Player 1", data['Player'].tolist())
            
            with col2:
                available_players = [p for p in data['Player'].tolist() if p != player1]
                player2 = st.selectbox("Select Player 2", available_players)
            
            if player1 and player2:
                #Get player data
                p1_data = data[data['Player'] == player1].iloc[0]
                p2_data = data[data['Player'] == player2].iloc[0]
                
                #Key statistics comparison
                key_stats = ['GAMES', 'RUNS', 'AVG', 'STRIKE RATE', 'WICKETS', 'ECONOMY RATE', 'TOTAL CATCHES']
                available_stats = [stat for stat in key_stats if stat in data.columns]
                
                comparison_data = []
                for stat in available_stats:
                    comparison_data.append({
                        'Statistic': stat,
                        player1: p1_data[stat],
                        player2: p2_data[stat]
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Radar chart comparison
                if len(available_stats) >= 3:
                    #Normalize values for radar chart
                    normalized_stats = []
                    for stat in available_stats[:6]:  # Limit to 6 stats for readability
                        max_val = data[stat].max()
                        min_val = data[stat].min()
                        if max_val != min_val:
                            p1_norm = (p1_data[stat] - min_val) / (max_val - min_val)
                            p2_norm = (p2_data[stat] - min_val) / (max_val - min_val)
                        else:
                            p1_norm = p2_norm = 0.5
                        normalized_stats.append((stat, p1_norm, p2_norm))
                    
                    #Create radar chart
                    categories = [stat[0] for stat in normalized_stats]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[stat[1] for stat in normalized_stats],
                        theta=categories,
                        fill='toself',
                        name=player1,
                        line_color='#1f4e79'
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[stat[2] for stat in normalized_stats],
                        theta=categories,
                        fill='toself',
                        name=player2,
                        line_color='#FDE725'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True,
                        title="Player Performance Comparison",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Player names not found in dataset")
    
    #Insights & Recommendations Tab
    elif analysis_type == "Insights & Recommendations":
        st.markdown('<h2 class="sub-header">Key Insights & Recommendations</h2>', unsafe_allow_html=True)
        
        #Generate insights based on the data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <h4>🎯 Selection Insights</h4>
            <ul>
            <li>Analyze the key performance indicators that correlate with selection</li>
            <li>Identify undervalued players with strong performance metrics</li>
            <li>Understand the selection criteria patterns</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if selection is not None:
                # Top performing non-selected players
                non_selected = data[data['SELECTION'] == 0]
                if len(non_selected) > 0 and 'AVG' in data.columns:
                    top_non_selected = non_selected.nlargest(5, 'AVG')
                    st.subheader("Top Non-Selected Players (by Batting Average)")
                    display_cols = ['Player', 'AVG', 'RUNS', 'STRIKE RATE']
                    available_cols = [col for col in display_cols if col in top_non_selected.columns]
                    st.dataframe(top_non_selected[available_cols], use_container_width=True)

                    top_non_selected_bwlav = non_selected.nsmallest(5, 'BOWL AVERAGE')
                    st.subheader("Top Non-Selected Players (by Bowling Average)")
                    display_cols = ['Player', 'RUNS CONCEDED', 'WICKETS', 'BOWL STRIKE RATE', 'BOWL AVERAGE']
                    available_cols = [col for col in display_cols if col in top_non_selected.columns]
                    st.dataframe(top_non_selected_bwlav[available_cols], use_container_width=True)

                    top_non_selected_bwlst = non_selected.nsmallest(5, 'BOWL STRIKE RATE')
                    st.subheader("Top Non-Selected Players (by Bowling Strike Rate)")
                    display_cols = ['Player', 'RUNS CONCEDED', 'WICKETS', 'BOWL STRIKE RATE', 'BOWL AVERAGE']
                    available_cols = [col for col in display_cols if col in top_non_selected.columns]
                    st.dataframe(top_non_selected_bwlst[available_cols], use_container_width=True)
        
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>📈 Performance Patterns</h4>
            <ul>
            <li>Cluster analysis reveals distinct player archetypes</li>
            <li>PCA shows the main dimensions of player performance</li>
            <li>XGBoost identifies the most predictive features</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance distribution
            if 'AVG' in data.columns and selection is not None:
                fig = px.box(data, x=['Not Selected' if s == 0 else 'Selected' for s in selection], 
                           y='AVG', title="Batting Average Distribution by Selection Status")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        #Recommendations
        st.markdown("---")
        st.subheader("Actionable Recommendations")
        
        recommendations = [
            "**Talent Identification**: Focus on players in high-performing clusters who weren't selected",
            "**Feature Engineering**: Use SHAP values to identify which metrics matter most for selection",
            "**Balanced Selection**: Consider both batting and bowling performance for well-rounded team composition",
            "**Performance Tracking**: Monitor key PCA components to track player development over time",
            "**Data-Driven Decisions**: Use model predictions to reinforce traditional scouting methods"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

else:
    st.info("🔍 **Getting Started:**")
    st.markdown("""
    **Option 1:** Upload your CSV file using the file uploader in the sidebar 
    
    **Option 2:** Place your `.csv` file in the same directory as this dashboard and refresh the page.
    
    **Required columns:** Your CSV should contain columns like `SELECTION`, `GAMES`, `RUNS`, `AVG`, `STRIKE RATE`, `WICKETS`, etc.
    """)
    
    # Show sample data format
    st.subheader("Expected Data Format")
    sample_data = pd.DataFrame({
        'Player': ['Player A', 'Player B', 'Player C'],
        'SELECTION': [1, 0, 1],
        'GAMES': [15, 12, 18],
        'RUNS': [450, 320, 520],
        'AVG': [35.5, 28.2, 41.3],
        'STRIKE RATE': [125.5, 118.2, 132.1],
        'WICKETS': [8, 15, 5],
        'ECONOMY RATE': [7.2, 6.8, 7.8]
    })
    st.dataframe(sample_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Durham County Junior Cricket Analytics Dashboard")