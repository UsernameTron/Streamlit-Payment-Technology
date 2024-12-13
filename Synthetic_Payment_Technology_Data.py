import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Load data from the local file
file_path = 'Synthetic_Payment_Technology_Data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
st.write("### Data Preview")
st.dataframe(df.head())

# Handle missing values
df['Rollout_Cost'] = df['Rollout_Cost'].fillna(df['Rollout_Cost'].median())
df['Revenue_Per_Machine_Last_12_Months'] = df['Revenue_Per_Machine_Last_12_Months'].fillna(df['Revenue_Per_Machine_Last_12_Months'].median())

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['Machines_Per_Room', 'Digital_Payment_Percentage', 'Rollout_Cost', 'Revenue_Per_Machine_Last_12_Months']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Feature Engineering
# Create new fields
df['Revenue_to_Cost_Ratio'] = df['Revenue_Per_Machine_Last_12_Months'] / df['Rollout_Cost']
df['Material_Efficiency'] = df['Materials_Required'] / df['Machines_Per_Room']
df['Digital_Percentage_Revenue'] = df['Digital_Payment_Percentage'] * df['Revenue_Per_Machine_Last_12_Months']

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Current_Technology_Type', 'Parent_Tier', 'Child_Tier', 'Setup_Configuration', 'Materials_Required']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Select features and target variable for Revenue prediction
features = ['Machines_Per_Room', 'Current_Technology_Type', 'Digital_Payment_Percentage', 'Parent_Tier', 'Revenue_to_Cost_Ratio', 'Material_Efficiency', 'Digital_Percentage_Revenue']
target = 'Revenue_Per_Machine_Last_12_Months'
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model for Revenue prediction
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions for Revenue
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
st.write(f'Revenue Model - Mean Squared Error: {mse}')
st.write(f'Revenue Model - R^2 Score: {r2}')
st.write(f'Revenue Model - Mean Absolute Error: {mae}')

# Visualize actual vs predicted values for Revenue
fig_revenue = px.scatter(
    x=y_test, y=y_pred,
    labels={'x': 'Actual Revenue Per Machine Last 12 Months', 'y': 'Predicted Revenue Per Machine Last 12 Months'},
    title='Actual vs Predicted Revenue Per Machine Last 12 Months'
)
fig_revenue.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')))
st.plotly_chart(fig_revenue)

# Feature importance plot
feature_importances = model.feature_importances_
features_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

fig_importance = px.bar(features_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
st.plotly_chart(fig_importance)

# Additional insights and visualizations
st.write("### Additional Insights")

# Exclude non-numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=[np.number])

# Correlation heatmap
corr = numeric_df.corr()
fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
st.plotly_chart(fig_corr)

# Distribution of Revenue Per Machine
fig_dist_revenue = px.histogram(df, x='Revenue_Per_Machine_Last_12_Months', nbins=50, title='Distribution of Revenue Per Machine Last 12 Months')
st.plotly_chart(fig_dist_revenue)

# Distribution of Rollout Cost
fig_dist_cost = px.histogram(df, x='Rollout_Cost', nbins=50, title='Distribution of Rollout Cost')
st.plotly_chart(fig_dist_cost)

# Bar graph highlighting high ROI configurations
high_roi_df = df[df['Revenue_to_Cost_Ratio'] > df['Revenue_to_Cost_Ratio'].median()]
fig_high_roi = px.bar(high_roi_df, x='Parent_Tier', y='Revenue_to_Cost_Ratio', color='Parent_Tier', title='High ROI Configurations by Parent Tier')
st.plotly_chart(fig_high_roi)

# Recommendation panel summarizing optimal configurations by Parent_Tier and predicted revenue
optimal_configurations = df.groupby('Parent_Tier').agg({
    'Revenue_Per_Machine_Last_12_Months': 'mean',
    'Revenue_to_Cost_Ratio': 'mean'
}).reset_index()
optimal_configurations = optimal_configurations.sort_values(by='Revenue_Per_Machine_Last_12_Months', ascending=False)

st.write("### Recommendation Panel")
st.dataframe(optimal_configurations)

# Scenario Simulation
st.write("### Scenario Simulation")

# Simulation tool for exploring outcomes of adjustments in Digital_Percentage or Setup_Configuration
digital_percentage = st.slider('Adjust Digital Percentage', min_value=0.0, max_value=1.0, step=0.01)
setup_configuration = st.selectbox('Select Setup Configuration', label_encoders['Setup_Configuration'].classes_)

# Simulate outcomes
simulation_df = df.copy()
simulation_df['Digital_Payment_Percentage'] = digital_percentage
simulation_df['Setup_Configuration'] = setup_configuration

# Ensure the selected setup configuration is valid
if setup_configuration in label_encoders['Setup_Configuration'].classes_:
    # Encode the new setup configuration
    simulation_df['Setup_Configuration'] = label_encoders['Setup_Configuration'].transform(simulation_df['Setup_Configuration'])
else:
    st.error("Selected setup configuration is not valid.")

# Predict revenue with the new configuration
simulation_X = simulation_df[features]
simulation_df['Predicted_Revenue'] = model.predict(simulation_X)

# Highlight recommended configurations for maximum ROI
recommended_configurations = simulation_df.groupby('Parent_Tier').agg({
    'Predicted_Revenue': 'mean',
    'Revenue_to_Cost_Ratio': 'mean'
}).reset_index()
recommended_configurations = recommended_configurations.sort_values(by='Predicted_Revenue', ascending=False)

st.write("### Recommended Configurations for Maximum ROI")
st.dataframe(recommended_configurations)