# Machine Learning Exam Study Guide
## IIC2115 - Programación como Herramienta para la Ingeniería

---

## SECTION 0: Core Distinctions (Start Here!)

### Supervised vs Unsupervised Learning

| Supervised Learning | Unsupervised Learning |
|---------------------|----------------------|
| **Has a target variable** (y) to predict | **No target variable** - discover structure |
| Examples: predict species, predict price | Examples: find customer groups, compress data |
| Split data: train/val/test | Usually use all data together |
| Metrics: accuracy, MSE, MAE | Metrics: silhouette score, inertia, variance explained |

### Regression vs Classification (Both are Supervised!)

**Regression (Regresión)**
- **Target variable:** Continuous number
- **Examples from exercises:**
  - E2: Predict PM2.5 value (23.4, 45.7, 102.3...)
  - E3: Predict median_house_value ($150,000, $325,000...)
- **Models:** Linear Regression, Random Forest Regressor, KNN Regressor
- **Metrics:** MSE, MAE, MAPE
- **Python check:** `y.dtype` will be `float64` or `int64` with many unique values

**Classification (Clasificación)**
- **Target variable:** Category/label
- **Examples from exercises:**
  - E1: Predict species (Adelie, Chinstrap, Gentoo)
  - E2: Predict environmental_risk (bajo, medio, alto, extremo)
  - E7: Predict resultado (abandono, continua_matriculado, graduado)
- **Models:** Logistic Regression, Random Forest Classifier, KNN Classifier, SVM
- **Metrics:** Accuracy, Balanced Accuracy
- **Python check:** `y.dtype` will be `object` or `int64` with few unique values (number of classes)

**Quick decision rule:**
```python
# If the target has few unique values → Classification
# If the target has many unique values → Regression

if df['target'].nunique() < 20:  # rough rule
    print("Probably CLASSIFICATION")
else:
    print("Probably REGRESSION")
```

### Dimensionality Reduction vs Clustering (Both are Unsupervised!)

**Dimensionality Reduction (Reducción de Dimensionalidad)**
- **Goal:** Transform many features → fewer features
- **Input:** Matrix X with n features
- **Output:** Matrix X_reduced with k features (k < n)
- **Methods:** PCA, t-SNE
- **Use when:** You have too many features (>10-15) or want to visualize in 2D/3D
- **Example:** 8 features → 2 principal components

```python
# BEFORE: X has 8 columns
# AFTER:  X_pca has 2 columns
# Same number of rows (same students/buildings/accidents)
```

**Clustering (Agrupamiento)**
- **Goal:** Group similar observations together
- **Input:** Matrix X with n features
- **Output:** Array of cluster labels [0, 1, 2, 0, 1, ...]
- **Methods:** K-Means, Agglomerative Clustering
- **Use when:** You want to find natural groups/profiles/patterns
- **Example:** 1000 students → 3 profiles (high performers, struggling, dropouts)

```python
# BEFORE: Each student has features but no group
# AFTER:  Each student assigned to cluster 0, 1, or 2
# Same number of rows, but now with a group label
```

**Can you combine them? YES!**

Three common approaches:
1. **Reduce THEN cluster:** Use PCA to compress features, then cluster on reduced space
2. **Cluster on original features:** Skip reduction if features are few and interpretable
3. **Do both and compare:** Try clustering on original vs reduced features (E4, E5, E7 often ask this)

**When to use each:**
- Images/high-dimensional data (>50 features): Reduce first (PCA → cluster)
- Tabular data (<20 features): Try both, compare
- The exam often asks you to **compare both approaches**

---

## SECTION 1: Essential ML Concepts & Methods

### Core Concepts (Conceptos Fundamentales)

#### Machine Learning Types

**Supervised Learning (Aprendizaje Supervisado)**
- Predict a known target variable
- Two subtypes:
  - **Regression:** predict continuous values (price, temperature, PM2.5)
  - **Classification:** predict categories (species, risk level, outcome)
- Requires train/validation/test split
- Examples: E1 (penguin species), E2 (PM2.5 & risk), E3 (house value), E7 (student outcome)

**Unsupervised Learning (Aprendizaje No Supervisado)**
- Discover structure without labels
- Two main tasks:
  - **Clustering:** find groups of similar observations
  - **Dimensionality reduction:** compress features while preserving information
- Usually uses all data together (no train/test split)
- Examples: E4 (housing profiles), E5 (art clustering), E6 (accident typologies)

#### Key Principles

**Overfitting (Sobreajuste)**
- Model learns training data too well, fails on new data
- High training performance, poor test performance
- **Symptoms:** Training accuracy 99%, Test accuracy 65%
- **Solution:** Simpler models, regularization, more data, early stopping

**Underfitting (Subajuste)**
- Model too simple to capture patterns
- Poor performance on both training and test data
- **Symptoms:** Training accuracy 60%, Test accuracy 58%
- **Solution:** More complex models, better features, remove constraints

**Generalization (Generalización)**
- Model's ability to perform well on unseen data
- The ultimate goal of ML models
- **Good sign:** Similar performance on train, validation, and test

**Data Leakage (Fuga de Datos)**
- Using information from validation/test during training
- **CRITICAL:** Fit scalers and encoders ONLY on training data
- **Example of leakage:**
```python
❌ DON'T:
scaler.fit(entire_dataset)  # LEAKAGE!

✅ DO:
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

#### Train/Validation/Test Split

```
Total Data (100%)
├── Training Set (60%): Learn model parameters
├── Validation Set (20%): Select hyperparameters & compare models
└── Test Set (20%): Final evaluation ONCE at the end
```

**Key Rules:**
1. **Never** use test set for decisions
2. Validation set can be used multiple times to compare models
3. Test set evaluates final selected model only once
4. Use `stratify` parameter to maintain class proportions in classification

**Why three sets?**
- **Training:** Model learns from this
- **Validation:** YOU learn which model/hyperparameters work best
- **Test:** Estimate real-world performance (don't peek until the end!)

---

### Model Comparison Tables

#### SUPERVISED MODELS (Predict y from X)

**For Regression Tasks** (continuous target)

| Model | Best For | Key Hyperparameter | Advantages | Disadvantages |
|-------|----------|-------------------|------------|---------------|
| **Linear Regression** | Linear relationships | None (or regularization) | Fast, interpretable, no tuning needed | Only linear relationships |
| **KNN Regressor** | Small datasets, simple patterns | `n_neighbors` (K) | No training needed, simple | Slow with large data, sensitive to scale |
| **Decision Tree Regressor** | Interpretability needed | `max_depth`, `min_samples_split` | Handles any data type, interpretable | Prone to overfitting |
| **Random Forest Regressor** | Tabular data, robust performance | `n_estimators`, `max_depth` | Reduces overfitting, robust, feature importance | Less interpretable, slower |
| **Gradient Boosting Regressor** | Best performance needed | `n_estimators`, `learning_rate` | Excellent performance | Overfitting risk, slower, needs tuning |

**For Classification Tasks** (categorical target)

| Model | Best For | Key Hyperparameter | Advantages | Disadvantages |
|-------|----------|-------------------|------------|---------------|
| **Logistic Regression** | Linear classification, baseline | `C` (regularization) | Fast, interpretable, probabilities | Only linear boundaries |
| **KNN Classifier** | Small datasets, simple patterns | `n_neighbors` (K) | No training needed, simple | Slow with large data, sensitive to scale |
| **Decision Tree Classifier** | Interpretability needed | `max_depth`, `min_samples_split` | Handles any data type, interpretable | Prone to overfitting |
| **Random Forest Classifier** | Tabular data, robust performance | `n_estimators`, `max_depth` | Reduces overfitting, robust | Less interpretable, slower |
| **SVM** | High-dimensional data | `C`, `kernel` | Effective, maximum margin | Slower training, less interpretable |
| **Gradient Boosting Classifier** | Competitions, best performance | `n_estimators`, `learning_rate` | Excellent performance | Overfitting risk, slower |

#### UNSUPERVISED METHODS (No target y)

**Dimensionality Reduction** (many features → few features)

| Method | Output | Key Parameter | When to Use |
|--------|--------|---------------|-------------|
| **PCA** | Linear projection to k components | `n_components` | Visualization, reducing features, linear relationships, **most common choice** |
| **t-SNE** | Non-linear projection (usually 2D) | `perplexity` | Visualization only, non-linear relationships, **don't use for clustering input** |

**Clustering** (group similar observations)

| Method | Task | Key Parameter | When to Use |
|--------|------|---------------|-------------|
| **K-Means** | Flat clustering | `n_clusters` (K) | Well-separated spherical clusters, **most common choice** |
| **Agglomerative Clustering** | Hierarchical clustering | `n_clusters`, `linkage` | Hierarchical structure, no need to predefine K |

---

### Most Common Model Choices (Based on Your Exercises)

**Safe choices for the exam:**

| Task | First Choice (baseline) | Second Choice (comparison) | Why |
|------|------------------------|----------------------------|-----|
| **Regression** | Linear Regression | Random Forest Regressor | Fast baseline + robust tree-based |
| **Classification** | Logistic Regression | Random Forest Classifier | Fast baseline + robust tree-based |
| **Clustering** | K-Means | Agglomerative (optional) | K-Means is almost always used |
| **Dimensionality Reduction** | PCA | t-SNE (for visualization only) | PCA is safer and more versatile |

**What exercises used:**
- **E1 (Penguins):** Classification → Logistic, KNN, Decision Tree, SVM
- **E2 (O3/PM2.5):** Regression (PM2.5, O3) → Linear, Random Forest
- **E2 (Environmental Risk):** Classification → Logistic, Random Forest
- **E3 (Housing):** Regression → Linear, Random Forest, Gradient Boosting
- **E4 (Housing Unsupervised):** PCA + K-Means
- **E5 (Art):** PCA, t-SNE, K-Means
- **E7 (Students):** PCA/t-SNE, K-Means → Random Forest Classifier

**The exercises usually say:**
- ✅ "entrene al menos dos modelos de regresión"
- ✅ "construya al menos 2 clasificadores"
- ✅ "evalúe distintas familias de modelos"

**They DON'T usually require specific models** — you choose, but:
- Must try at least 2 different types
- Must compare on validation
- Must select best and evaluate on test

---

### Key Metrics (Métricas)

#### Classification Metrics

**Accuracy (Exactitud)**
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
```
- Proportion of correct predictions: `(TP + TN) / Total`
- Range: [0, 1], higher is better
- **Problem:** Poor with imbalanced classes
- **Use when:** Classes are balanced (roughly equal counts)

**Balanced Accuracy**
```python
from sklearn.metrics import balanced_accuracy_score
balanced_acc = balanced_accuracy_score(y_true, y_pred)
```
- Average recall per class
- Range: [0, 1], higher is better
- **Use when:** Classes are imbalanced (e.g., 90% class A, 10% class B)
- **Most exercises use this:** E2, E7

#### Regression Metrics

**MSE (Mean Squared Error / Error Cuadrático Medio)**
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```
- Penalizes large errors heavily
- Range: [0, ∞], lower is better
- Units: squared target units
- **Use when:** Large errors are particularly bad

**MAE (Mean Absolute Error / Error Absoluto Medio)**
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```
- Average absolute difference
- Range: [0, ∞], lower is better
- Units: same as target
- More robust to outliers than MSE
- **Use when:** You want interpretable error in original units

**MAPE (Mean Absolute Percentage Error / Error Porcentual Absoluto Medio)**
```python
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_true, y_pred)
```
- Percentage error, interpretable
- Range: [0, ∞], lower is better (usually report as percentage)
- **Problem:** Undefined when y_true = 0
- **Use when:** You want to compare across different scales

**Which to use?**
- Exercises E2, E3 often ask for **all three (MSE, MAE, MAPE)**
- If you can only pick one: **MAE** (most interpretable)

#### Clustering Metrics

**Silhouette Score**
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, cluster_labels)
```
- Range: [-1, 1], higher is better
- Measures cluster separation and cohesion
- **Use to:** Compare different values of K
- **Exercises:** E4, E5, E6, E7 use this to select K

**Inertia (within-cluster sum of squares)**
```python
inertia = kmeans.inertia_
```
- Range: [0, ∞], lower is better
- Always decreases with more clusters
- **Use to:** Create elbow plot to select K
- **Note:** Don't just pick lowest inertia (that would be K=n_samples)

**How to select K:**
1. Try K from 2 to 10
2. Plot elbow curve (inertia vs K)
3. Plot silhouette score vs K
4. Pick K where:
   - Elbow bends (diminishing returns)
   - Silhouette is high
   - Clusters are interpretable

#### Dimensionality Reduction Metrics

**Explained Variance Ratio (for PCA)**
```python
pca.explained_variance_ratio_
```
- Proportion of variance captured by each component
- Sum of first k components = total variance explained
- **Use to:** Decide how many components to keep
- **Rule of thumb:** Keep components until 80-90% variance explained

---

### Spanish Technical Vocabulary

| English | Spanish |
|---------|---------|
| Training set | Conjunto de entrenamiento |
| Validation set | Conjunto de validación |
| Test set | Conjunto de prueba/test |
| Features | Características / variables |
| Target variable | Variable objetivo |
| Overfitting | Sobreajuste |
| Underfitting | Subajuste |
| Data leakage | Fuga de datos |
| Hyperparameter | Hiperparámetro |
| Clustering | Agrupamiento / clustering |
| Dimensionality reduction | Reducción de dimensionalidad |
| Supervised learning | Aprendizaje supervisado |
| Unsupervised learning | Aprendizaje no supervisado |
| Accuracy | Exactitud |
| Precision | Precisión |
| Recall | Sensibilidad / recall |
| Regression | Regresión |
| Classification | Clasificación |
| Derived variables | Variables derivadas |

---

## SECTION 2: Modular Code Blocks

### 1. Data Loading & Inspection

**Load CSV data**
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('filename.csv')

# Quick inspection
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # Missing values per column

# Check for class imbalance (for classification)
print(df['target_column'].value_counts())
print(df['target_column'].value_counts(normalize=True))  # proportions
```

**Key tips:**
- Check data types: numerical should be float/int, categorical should be object
- Identify missing values early
- Check for class imbalance in classification tasks
- Look for outliers in `.describe()` (min/max values)

---

### 2. Handling Missing Values (Valores Faltantes)

**Strategy 1: Fill numerical missing values with mean**
```python
# Fill each numeric column with its mean
numeric_cols = ['col1', 'col2', 'col3']
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)
```

**Strategy 2: Fill numerical with median (more robust to outliers)**
```python
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)
```

**Strategy 3: Drop rows with categorical missing values**
```python
# Drop rows where categorical columns have NaN
df.dropna(subset=['categorical_col1', 'categorical_col2'], inplace=True)
```

**Strategy 4: Fill categorical with mode**
```python
df['categorical_col'].fillna(df['categorical_col'].mode()[0], inplace=True)
```

**Key tips:**
- **E2 approach:** Fill numerical with mean, drop categorical rows
- Decide before splitting train/val/test
- For supervised: can fill within each split to avoid leakage (use train mean for train/val/test)
- Document your decision

---

### 3. Encoding Categorical Variables

**Label Encoding (for target variable in classification)**
```python
from sklearn.preprocessing import LabelEncoder

# Encode target variable
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# To see mapping
print(dict(zip(label_encoder.classes_, 
               label_encoder.transform(label_encoder.classes_))))

# To decode back later (for interpretation)
decoded = label_encoder.inverse_transform([0, 1, 2])
```

**One-Hot Encoding (for categorical features)**
```python
# For categorical features with no ordinal relationship
df = pd.get_dummies(df, columns=['categorical_col1', 'categorical_col2'], 
                    drop_first=False)

# Alternative with explicit control
df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean', drop_first=True)
```

**When to encode:**
- ✅ **Before splitting** if you want same columns in train/val/test (safest for one-hot)
- ✅ **After splitting** if you want to avoid any potential leakage
  - For one-hot after split: use `pd.get_dummies()` on each set separately, then align columns

**Key tips:**
- LabelEncoder for **target variable only**
- One-hot encoding for **categorical features**
- `drop_first=True` can reduce dimensionality but not always necessary
- Be careful with one-hot: if you do it after split, ensure all sets have same columns

---

### 4. Feature Engineering (Creating Derived Variables)

**Ratios (per capita / per unit metrics)**
```python
# E3, E4, E7 examples
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# E7 example
df['approval_rate'] = df['asignaturas_aprobadas'] / df['asignaturas_inscritas']
df['exam_completion_rate'] = df['evaluaciones_realizadas'] / df['asignaturas_inscritas']
```

**Binary flags (indicator variables)**
```python
# E4 example
df['is_near_ocean'] = (df['ocean_proximity'] == 'NEAR OCEAN').astype(int)

# E7 example
df['has_convalidaciones'] = (df['asignaturas_convalidadas'] > 0).astype(int)
df['perfect_attendance'] = (df['asignaturas_sin_evaluacion'] == 0).astype(int)
```

**Binning continuous variables**
```python
# E3 example: Age groups
df['age_category'] = pd.cut(df['housing_median_age'], 
                             bins=[0, 10, 25, 50], 
                             labels=['new', 'medium', 'old'])

# Then one-hot encode
df = pd.get_dummies(df, columns=['age_category'], drop_first=True)
```

**Temporal features**
```python
# E2 example: Convert date parts
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Day'] = df['Day'].astype(int)

# Create cyclical features for month (optional advanced)
df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
```

**Interaction terms**
```python
# Multiply two features to capture interaction
df['income_x_age'] = df['median_income'] * df['housing_median_age']
```

**Key tips:**
- Create at least 3 derived variables for exercises
- Choose variables that make domain sense (rooms per household, not random ratios)
- **Handle division by zero:**
  ```python
  df['ratio'] = df['numerator'] / df['denominator'].replace(0, np.nan)
  df['ratio'].fillna(0, inplace=True)  # or drop these rows
  ```

---

### 5. Train/Validation/Test Split

**Two-step split for 60/20/20 split**
```python
from sklearn.model_selection import train_test_split

# First split: 60% train, 40% temp
X = df[feature_columns]
y = df['target']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.40, 
    random_state=42,
    stratify=y  # For classification: keeps class proportions
)

# Second split: split temp into 50/50 → 20% val, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp  # For classification
)

print(f'Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)')
print(f'Val size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)')
print(f'Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)')
```

**Alternative: using DataFrame splitting**
```python
train_set, temp_set = train_test_split(
    df, test_size=0.40, random_state=42, stratify=df['target']
)
val_set, test_set = train_test_split(
    temp_set, test_size=0.50, random_state=42, stratify=temp_set['target']
)

# Then extract X and y from each
X_train = train_set[feature_columns]
y_train = train_set['target']
# ... and so on
```

**Key tips:**
- Use `stratify` for classification to maintain class balance
- Don't use `stratify` for regression
- `random_state=42` for reproducibility
- **Test set is sacred:** don't touch until final evaluation!

---

### 6. Scaling/Normalization

**StandardScaler (most common - zero mean, unit variance)**
```python
from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Fit ONLY on training data
scaler.fit(X_train)

# Transform all three sets using the SAME scaler
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# If working with DataFrames and want to keep column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
```

**When scaling is needed:**
- ✅ KNN (distance-based)
- ✅ SVM
- ✅ Logistic Regression (helps convergence)
- ✅ PCA (essential!)
- ✅ K-Means clustering
- ❌ Decision Trees / Random Forest (not needed, but doesn't hurt)

**CRITICAL RULE: No Data Leakage!**
```python
❌ DON'T:
scaler.fit(entire_dataset)  # Uses info from test set!

✅ DO:
scaler.fit(X_train)  # Only learns from training
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

---

### 7. Training Supervised Models

**Linear Regression**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_val_pred = model.predict(X_val_scaled)
```

**Logistic Regression (Classification)**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_val_pred = model.predict(X_val_scaled)
```

**KNN (Classifier or Regressor)**
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# For classification
model = KNeighborsClassifier(n_neighbors=5)

# For regression
# model = KNeighborsRegressor(n_neighbors=5)

model.fit(X_train_scaled, y_train)
y_val_pred = model.predict(X_val_scaled)
```

**Decision Tree**
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# For classification
model = DecisionTreeClassifier(max_depth=5, random_state=42)

# For regression
# model = DecisionTreeRegressor(max_depth=5, random_state=42)

model.fit(X_train_scaled, y_train)
y_val_pred = model.predict(X_val_scaled)
```

**Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# For classification
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42
)

# For regression
# model = RandomForestRegressor(
#     n_estimators=100, 
#     max_depth=10, 
#     random_state=42
# )

model.fit(X_train_scaled, y_train)
y_val_pred = model.predict(X_val_scaled)
```

**SVM (Support Vector Machine)**
```python
from sklearn.svm import SVC, SVR

# For classification
model = SVC(kernel='rbf', C=1.0, random_state=42)

# For regression
# model = SVR(kernel='rbf', C=1.0)

model.fit(X_train_scaled, y_train)
y_val_pred = model.predict(X_val_scaled)
```

**Gradient Boosting**
```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# For classification
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# For regression
# model = GradientBoostingRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=42
# )

model.fit(X_train_scaled, y_train)
y_val_pred = model.predict(X_val_scaled)
```

---

### 8. Evaluating Models

**For Classification**
```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# On validation set
y_val_pred = model.predict(X_val_scaled)

acc = accuracy_score(y_val, y_val_pred)
balanced_acc = balanced_accuracy_score(y_val, y_val_pred)

print(f'Validation Accuracy: {acc:.3f}')
print(f'Validation Balanced Accuracy: {balanced_acc:.3f}')
```

**For Regression**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

y_val_pred = model.predict(X_val_scaled)

mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
mape = mean_absolute_percentage_error(y_val, y_val_pred)

print(f'Validation MSE: {mse:.3f}')
print(f'Validation MAE: {mae:.3f}')
print(f'Validation MAPE: {mape:.3f}')
```

---

### 9. Hyperparameter Tuning with Validation

**Example: Tuning K for KNN**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score

k_values = [1, 3, 5, 7, 9, 11, 15]
results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    
    y_val_pred = model.predict(X_val_scaled)
    val_score = balanced_accuracy_score(y_val, y_val_pred)
    
    results.append({'K': k, 'balanced_accuracy': val_score})

results_df = pd.DataFrame(results)
print(results_df)

# Select best K
best_k = results_df.loc[results_df['balanced_accuracy'].idxmax(), 'K']
print(f'\nBest K: {best_k}')
```

**Example: Tuning max_depth for Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

depths = [3, 5, 10, 15, 20, None]
results = []

for depth in depths:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_val_pred = model.predict(X_val_scaled)
    val_score = balanced_accuracy_score(y_val, y_val_pred)
    
    results.append({'max_depth': depth, 'balanced_accuracy': val_score})

results_df = pd.DataFrame(results)
print(results_df)

best_depth = results_df.loc[results_df['balanced_accuracy'].idxmax(), 'max_depth']
```

**Key tips:**
- Try multiple values for each hyperparameter
- Always evaluate on validation set (never test!)
- Can tune multiple hyperparameters together (grid search)
- Keep a table of results to compare

---

### 10. Model Comparison

**Compare multiple model families**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

# Define candidate models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation
    y_val_pred = model.predict(X_val_scaled)
    val_score = balanced_accuracy_score(y_val, y_val_pred)
    
    results.append({
        'Model': name,
        'Validation Balanced Accuracy': val_score
    })

results_df = pd.DataFrame(results)
print(results_df)

# Select best model
best_model_name = results_df.loc[results_df['Validation Balanced Accuracy'].idxmax(), 'Model']
print(f'\nBest Model: {best_model_name}')
```

---

### 11. Final Evaluation on Test Set

**Train final model and evaluate ONCE on test**
```python
from sklearn.base import clone

# Get the best model from comparison
best_model = models[best_model_name]

# Clone and retrain (optional: could use already trained model)
final_model = clone(best_model)
final_model.fit(X_train_scaled, y_train)

# Evaluate on test set (ONLY ONCE!)
y_test_pred = final_model.predict(X_test_scaled)
test_score = balanced_accuracy_score(y_test, y_test_pred)

print(f'\n=== FINAL EVALUATION ===')
print(f'Test Balanced Accuracy: {test_score:.3f}')
```

**Key tips:**
- Only evaluate on test ONCE at the very end
- If test score is much worse than validation: possible overfitting to validation
- If test score is similar to validation: good generalization!
- **Never** go back and retrain after seeing test results

---

### 12. PCA (Principal Component Analysis)

**Apply PCA for dimensionality reduction**
```python
from sklearn.decomposition import PCA

# Decide number of components (e.g., 2 for visualization)
pca = PCA(n_components=2, random_state=42)

# Fit on training data
pca.fit(X_train_scaled)

# Transform all sets
X_train_pca = pca.transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Convert back to DataFrame (optional)
pca_columns = [f'PC{i+1}' for i in range(pca.n_components)]
X_train_pca = pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)

# Check explained variance
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
print(f'Total variance explained: {pca.explained_variance_ratio_.sum():.2%}')
```

**Choosing number of components**
```python
# Try different numbers and plot cumulative variance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca_full = PCA()
pca_full.fit(X_train_scaled)

# Cumulative variance explained
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum)+1), cumsum, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.80, color='r', linestyle='--', label='80% threshold')
plt.axhline(y=0.90, color='g', linestyle='--', label='90% threshold')
plt.legend()
plt.grid()
plt.show()

# Find number of components for 90% variance
n_components_90 = np.argmax(cumsum >= 0.90) + 1
print(f'Components needed for 90% variance: {n_components_90}')
```

**Interpreting principal components**
```python
# Get feature loadings (contributions of original features to PCs)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components)],
    index=X_train.columns
)

print(loadings)

# Visualize loadings
loadings.plot(kind='bar', figsize=(10, 6))
plt.title('Feature Contributions to Principal Components')
plt.ylabel('Loading')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

**Key tips:**
- **Always scale data before PCA** (essential!)
- Fit PCA only on training data
- For visualization: use 2-3 components
- For clustering/modeling: keep enough components for 80-90% variance
- PCA components are not directly interpretable but you can analyze loadings

---

### 13. t-SNE (for visualization only)

**Apply t-SNE for 2D visualization**
```python
from sklearn.manifold import TSNE

# t-SNE for visualization (usually 2 components)
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    random_state=42
)

# Fit and transform (t-SNE doesn't separate fit/transform)
# Only use on data you want to visualize (e.g., train+val, or all data)
X_tsne = tsne.fit_transform(X_scaled)

# Convert to DataFrame
tsne_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
```

**Key differences from PCA:**
- t-SNE is **non-linear** (better for complex patterns)
- t-SNE is **only for visualization** (can't transform new data)
- t-SNE is **slower** than PCA
- **Don't use t-SNE output as input to clustering** (use PCA instead)

**When to use:**
- ✅ Visualizing high-dimensional data in 2D
- ✅ Comparing to PCA visualization
- ❌ As input to supervised models
- ❌ As input to clustering (use PCA instead)

---

### 14. K-Means Clustering

**Basic K-Means**
```python
from sklearn.cluster import KMeans

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    random_state=42
)

# Fit on data (use scaled data!)
kmeans.fit(X_scaled)

# Get cluster labels
cluster_labels = kmeans.labels_

# Add to DataFrame
df['cluster'] = cluster_labels
```

**Elbow method to select K**
```python
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.xticks(K_range)
plt.grid()
plt.show()
```

**Silhouette method to select K**
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.xticks(K_range)
plt.grid()
plt.show()

# Select best K
best_k = K_range[np.argmax(silhouette_scores)]
print(f'Best K by silhouette: {best_k}')
```

**Key tips:**
- Always scale data before K-Means
- Try K from 2 to 10
- Use both elbow and silhouette to decide
- Consider interpretability: can you explain each cluster?

---

### 15. Agglomerative (Hierarchical) Clustering

**Basic Agglomerative Clustering**
```python
from sklearn.cluster import AgglomerativeClustering

# Create model
agg_clustering = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

# Fit and get labels
cluster_labels = agg_clustering.fit_predict(X_scaled)

df['cluster'] = cluster_labels
```

**Linkage methods:**
- `'ward'`: minimizes variance (most common)
- `'complete'`: maximum distance
- `'average'`: average distance
- `'single'`: minimum distance

**Key tips:**
- Similar to K-Means but builds hierarchy
- Also needs scaled data
- Can visualize with dendrogram (advanced)
- Usually K-Means is preferred for simplicity

---

### 16. Characterizing Clusters

**Get cluster statistics**
```python
# Group by cluster and compute means
cluster_summary = df.groupby('cluster')[feature_columns].mean()
print(cluster_summary)

# Count observations per cluster
print(df['cluster'].value_counts())

# Get proportions
print(df['cluster'].value_counts(normalize=True))
```

**Visualize clusters in 2D (using PCA)**
```python
import matplotlib.pyplot as plt

# Assuming you have X_pca (2D PCA) and cluster labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], 
    X_pca[:, 1], 
    c=cluster_labels, 
    cmap='viridis',
    alpha=0.6
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters in PCA Space')
plt.colorbar(scatter, label='Cluster')
plt.show()
```

**Compare clusters on a target variable (for interpretation)**
```python
# E4, E7 examples: compare median_house_value across clusters
cluster_comparison = df.groupby('cluster')['median_house_value'].agg(['mean', 'median', 'std'])
print(cluster_comparison)

# Boxplot
df.boxplot(column='median_house_value', by='cluster', figsize=(8, 5))
plt.title('House Value by Cluster')
plt.suptitle('')
plt.ylabel('Median House Value')
plt.show()
```

**Key tips:**
- Characterize clusters using **original features** (not PCA components)
- Give each cluster a descriptive name
- Check if clusters align with target variable (if supervised task follows)

---

### 17. Error Analysis by Group

**E7 scenario: Evaluate model performance per cluster**
```python
from sklearn.metrics import balanced_accuracy_score

# Assuming you have cluster labels for test set
test_df = X_test.copy()
test_df['cluster'] = cluster_labels_test
test_df['y_true'] = y_test
test_df['y_pred'] = y_test_pred

# Overall performance
overall_acc = balanced_accuracy_score(y_test, y_test_pred)
print(f'Overall Test Accuracy: {overall_acc:.3f}')

# Performance per cluster
for cluster_id in sorted(test_df['cluster'].unique()):
    cluster_mask = test_df['cluster'] == cluster_id
    y_true_cluster = test_df.loc[cluster_mask, 'y_true']
    y_pred_cluster = test_df.loc[cluster_mask, 'y_pred']
    
    acc_cluster = balanced_accuracy_score(y_true_cluster, y_pred_cluster)
    print(f'Cluster {cluster_id}: {acc_cluster:.3f} (n={cluster_mask.sum()})')
```

**Compare errors across categories**
```python
# E3 example: errors by ocean_proximity
test_df['error'] = np.abs(y_test - y_test_pred)

error_by_category = test_df.groupby('ocean_proximity')['error'].agg(['mean', 'median'])
print(error_by_category)
```

---

### 18. Visualization

**Confusion Matrix (for classification)**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.show()
```

**Feature Importance (for tree-based models)**
```python
# For Random Forest, Gradient Boosting, Decision Tree
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df.head(10))

# Plot
plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df['feature'][:10], 
         feature_importance_df['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()
```

**Distribution plots**
```python
# Compare distributions across clusters
for feature in ['median_income', 'housing_median_age']:
    df.boxplot(column=feature, by='cluster', figsize=(8, 5))
    plt.title(f'{feature} by Cluster')
    plt.suptitle('')
    plt.show()
```

---

### 19. Common Pitfalls to Avoid

**❌ DON'T:**

1. **Fit scaler/encoder on entire dataset**
   ```python
   scaler.fit(entire_dataset)  # LEAKAGE!
   ```

2. **Use test set for hyperparameter tuning**
   ```python
   # Trying different K values and checking test accuracy
   for k in k_values:
       model = KNN(k)
       model.fit(X_train, y_train)
       test_acc = accuracy_score(y_test, model.predict(X_test))  # WRONG!
   ```

3. **Include target variable in clustering**
   ```python
   # E7: DON'T include 'resultado' in desempeño clustering
   kmeans.fit(df[features + ['resultado']])  # WRONG!
   ```

4. **Use t-SNE output for clustering/modeling**
   ```python
   X_tsne = tsne.fit_transform(X)
   kmeans.fit(X_tsne)  # BAD PRACTICE (use PCA instead)
   ```

5. **Make decisions after seeing test results**
   - Evaluate test → "hmm, let me try a different model" → WRONG!

**✅ DO:**

1. **Fit transformations only on training data**
   ```python
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_val_scaled = scaler.transform(X_val)
   X_test_scaled = scaler.transform(X_test)
   ```

2. **Use validation for all decisions**
   ```python
   for k in k_values:
       model = KNN(k)
       model.fit(X_train, y_train)
       val_acc = accuracy_score(y_val, model.predict(X_val))  # CORRECT
   ```

3. **Separate clustering features from target**
   ```python
   # E7: cluster only on desempeño features
   clustering_features = ['asignaturas_aprobadas', 'promedio_notas', ...]
   kmeans.fit(df[clustering_features])  # CORRECT
   ```

4. **Use PCA for clustering input**
   ```python
   X_pca = pca.fit_transform(X_scaled)
   kmeans.fit(X_pca)  # CORRECT
   ```

5. **Test evaluation is final**
   - Make all decisions with validation
   - Evaluate test once at the end
   - Report test result and conclude

---

## SECTION 3: E7-Style Workflow Checklist

This section provides a step-by-step guide for **integrated exercises** that combine supervised and unsupervised learning (like E7).

### Phase 1: Data Preparation

**Goal:** Create two comparable feature matrices

**Steps:**
1. Load data and inspect
2. Handle missing values
3. Encode categorical variables
4. Create **at least 3 derived variables** from performance data
   - Ratios (approval rate, completion rate)
   - Binary flags (has_X, perfect_Y)
   - Normalized metrics (per capita, per unit)
5. Build two matrices:
   - **Matrix Ingreso:** Only admission + economic context features
   - **Matrix Actualizada:** Ingreso + performance (original + derived)
6. Verify both matrices have same number of rows

**Key tip:** Don't include target variable (e.g., `resultado`) in either matrix

---

### Phase 2: Unsupervised Analysis (Profiles)

**Goal:** Find performance profiles using clustering

**Steps:**

1. **Select clustering features**
   - Use ONLY performance variables (original + derived)
   - Exclude: target, admission info, economic context
   
2. **Scale the clustering features**
   ```python
   scaler = StandardScaler()
   X_clustering_scaled = scaler.fit_transform(df[clustering_features])
   ```

3. **Apply dimensionality reduction (optional but common)**
   ```python
   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X_clustering_scaled)
   
   # Analyze explained variance
   print(f'Variance explained: {pca.explained_variance_ratio_.sum():.2%}')
   
   # Visualize in 2D
   plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
   ```

4. **Cluster selection**
   - Try K from 2 to 10
   - Compute inertia and silhouette scores
   - Plot elbow and silhouette curves
   - Select best K based on metrics + interpretability

5. **Fit final clustering**
   ```python
   best_k = 3  # from selection above
   kmeans = KMeans(n_clusters=best_k, random_state=42)
   cluster_labels = kmeans.fit_predict(X_clustering_scaled)
   df['cluster'] = cluster_labels
   ```

6. **Characterize profiles**
   - Compute mean/median of features per cluster
   - Visualize distributions
   - Assign descriptive names (e.g., "High Performers", "At Risk")
   - Check distribution of target variable across clusters

7. **Compare clustering on original vs PCA features** (if asked)
   - Cluster on `X_clustering_scaled` vs `X_pca`
   - Compare silhouette scores
   - Select one approach and justify

**Key tip:** Clusters are created using performance only, but can be interpreted using all variables (including target for interpretation, not for construction)

---

### Phase 3: Supervised Prediction

**Goal:** Compare prediction with/without performance information

**Steps:**

1. **Prepare target and splits**
   ```python
   y = df['resultado']  # or whatever target
   
   # Encode if categorical
   from sklearn.preprocessing import LabelEncoder
   le = LabelEncoder()
   y = le.fit_transform(y)
   ```

2. **Split data for Matrix Ingreso**
   ```python
   X_ingreso = df[ingreso_features]
   
   X_train_ing, X_temp, y_train, y_temp = train_test_split(
       X_ingreso, y, test_size=0.4, random_state=42, stratify=y
   )
   X_val_ing, X_test_ing, y_val, y_test = train_test_split(
       X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
   )
   ```

3. **Split data for Matrix Actualizada** (same random_state!)
   ```python
   X_actual = df[ingreso_features + performance_features]
   
   X_train_act, X_temp, y_train, y_temp = train_test_split(
       X_actual, y, test_size=0.4, random_state=42, stratify=y
   )
   X_val_act, X_test_act, y_val, y_test = train_test_split(
       X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
   )
   ```

4. **Scale both matrices**
   ```python
   scaler_ing = StandardScaler()
   X_train_ing_sc = scaler_ing.fit_transform(X_train_ing)
   X_val_ing_sc = scaler_ing.transform(X_val_ing)
   X_test_ing_sc = scaler_ing.transform(X_test_ing)
   
   scaler_act = StandardScaler()
   X_train_act_sc = scaler_act.fit_transform(X_train_act)
   X_val_act_sc = scaler_act.transform(X_val_act)
   X_test_act_sc = scaler_act.transform(X_test_act)
   ```

5. **Train models on both matrices**
   
   For **Matrix Ingreso:**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.linear_model import LogisticRegression
   
   models_ing = {
       'Logistic': LogisticRegression(max_iter=1000, random_state=42),
       'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
   }
   
   results_ing = []
   for name, model in models_ing.items():
       model.fit(X_train_ing_sc, y_train)
       y_val_pred = model.predict(X_val_ing_sc)
       val_acc = balanced_accuracy_score(y_val, y_val_pred)
       results_ing.append({'Model': name, 'Val_Acc': val_acc})
   
   results_ing_df = pd.DataFrame(results_ing)
   best_model_ing_name = results_ing_df.loc[results_ing_df['Val_Acc'].idxmax(), 'Model']
   ```
   
   For **Matrix Actualizada:**
   ```python
   models_act = {
       'Logistic': LogisticRegression(max_iter=1000, random_state=42),
       'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
   }
   
   results_act = []
   for name, model in models_act.items():
       model.fit(X_train_act_sc, y_train)
       y_val_pred = model.predict(X_val_act_sc)
       val_acc = balanced_accuracy_score(y_val, y_val_pred)
       results_act.append({'Model': name, 'Val_Acc': val_acc})
   
   results_act_df = pd.DataFrame(results_act)
   best_model_act_name = results_act_df.loc[results_act_df['Val_Acc'].idxmax(), 'Model']
   ```

6. **Optional: Same model, both matrices (isolate feature contribution)**
   ```python
   # Fix model family (e.g., Random Forest)
   model_ing = RandomForestClassifier(n_estimators=100, random_state=42)
   model_ing.fit(X_train_ing_sc, y_train)
   
   model_act = RandomForestClassifier(n_estimators=100, random_state=42)
   model_act.fit(X_train_act_sc, y_train)
   
   val_acc_ing = balanced_accuracy_score(y_val, model_ing.predict(X_val_ing_sc))
   val_acc_act = balanced_accuracy_score(y_val, model_act.predict(X_val_act_sc))
   
   print(f'Same model (RF):')
   print(f'  Ingreso only: {val_acc_ing:.3f}')
   print(f'  Actualizada: {val_acc_act:.3f}')
   print(f'  Improvement: {val_acc_act - val_acc_ing:.3f}')
   ```

7. **Final evaluation on test**
   ```python
   # Best models from validation
   final_model_ing = models_ing[best_model_ing_name]
   final_model_act = models_act[best_model_act_name]
   
   # Retrain (or use already trained)
   final_model_ing.fit(X_train_ing_sc, y_train)
   final_model_act.fit(X_train_act_sc, y_train)
   
   # Test evaluation
   y_test_pred_ing = final_model_ing.predict(X_test_ing_sc)
   y_test_pred_act = final_model_act.predict(X_test_act_sc)
   
   test_acc_ing = balanced_accuracy_score(y_test, y_test_pred_ing)
   test_acc_act = balanced_accuracy_score(y_test, y_test_pred_act)
   
   print(f'\n=== FINAL TEST RESULTS ===')
   print(f'Ingreso only: {test_acc_ing:.3f}')
   print(f'Actualizada (with performance): {test_acc_act:.3f}')
   print(f'Improvement: {test_acc_act - test_acc_ing:.3f}')
   ```

---

### Phase 4: Performance Analysis by Profile

**Goal:** Check if prediction quality/improvement varies by cluster

**Steps:**

1. **Assign clusters to test set**
   ```python
   # You need cluster labels for test set
   # If you clustered on full dataset earlier:
   test_clusters = df.loc[X_test_ing.index, 'cluster']
   
   # If you need to assign test set to existing clusters:
   # (scale test performance features, then predict cluster)
   # test_clusters = kmeans.predict(scaler_clustering.transform(X_test_performance))
   ```

2. **Evaluate overall and per-cluster**
   ```python
   from sklearn.metrics import balanced_accuracy_score
   
   # Create results DataFrame
   test_results = pd.DataFrame({
       'cluster': test_clusters,
       'y_true': y_test,
       'y_pred_ing': y_test_pred_ing,
       'y_pred_act': y_test_pred_act
   })
   
   # Overall
   print('=== Overall Test Performance ===')
   print(f'Ingreso: {test_acc_ing:.3f}')
   print(f'Actualizada: {test_acc_act:.3f}')
   
   # Per cluster
   print('\n=== Performance by Cluster ===')
   for cluster_id in sorted(test_results['cluster'].unique()):
       mask = test_results['cluster'] == cluster_id
       
       acc_ing = balanced_accuracy_score(
           test_results.loc[mask, 'y_true'],
           test_results.loc[mask, 'y_pred_ing']
       )
       acc_act = balanced_accuracy_score(
           test_results.loc[mask, 'y_true'],
           test_results.loc[mask, 'y_pred_act']
       )
       improvement = acc_act - acc_ing
       
       print(f'Cluster {cluster_id} (n={mask.sum()}):')
       print(f'  Ingreso: {acc_ing:.3f}')
       print(f'  Actualizada: {acc_act:.3f}')
       print(f'  Improvement: {improvement:+.3f}')
   ```

3. **Visualize comparison**
   ```python
   import matplotlib.pyplot as plt
   
   cluster_comparison = []
   for cluster_id in sorted(test_results['cluster'].unique()):
       mask = test_results['cluster'] == cluster_id
       acc_ing = balanced_accuracy_score(
           test_results.loc[mask, 'y_true'],
           test_results.loc[mask, 'y_pred_ing']
       )
       acc_act = balanced_accuracy_score(
           test_results.loc[mask, 'y_true'],
           test_results.loc[mask, 'y_pred_act']
       )
       cluster_comparison.append({
           'Cluster': f'Cluster {cluster_id}',
           'Ingreso': acc_ing,
           'Actualizada': acc_act
       })
   
   comp_df = pd.DataFrame(cluster_comparison)
   
   comp_df.plot(x='Cluster', y=['Ingreso', 'Actualizada'], kind='bar', figsize=(8, 5))
   plt.ylabel('Balanced Accuracy')
   plt.title('Model Performance by Cluster')
   plt.xticks(rotation=0)
   plt.legend(['Ingreso Only', 'With Performance Data'])
   plt.tight_layout()
   plt.show()
   ```

4. **Interpret findings**
   - Which clusters benefit most from performance data?
   - Which clusters are hardest to predict?
   - Does cluster characterization explain prediction difficulty?

---

### Phase 5: Conclusions and Limitations

**What to include:**

1. **Summary of findings**
   - How many profiles found?
   - What characterizes each profile?
   - How much does performance data improve prediction?

2. **Profile-specific insights**
   - Is improvement uniform across profiles?
   - Which profiles are easier/harder to predict?

3. **Limitations**
   - Data limitations (missing values, imbalanced classes, small sample)
   - Model limitations (linear boundaries, overfitting risk)
   - Clustering stability (try different random_state?)
   - Generalization concerns (specific institution, time period)

4. **Possible improvements**
   - More features (social, geographical)
   - More sophisticated models
   - Temporal analysis (track changes over semesters)
   - Ensemble methods

**Example conclusion:**
```
"Se identificaron 3 perfiles de desempeño: alto rendimiento (30%), 
rendimiento medio (50%) y en riesgo (20%). La incorporación de variables 
de desempeño mejoró la predicción del resultado en 15 puntos porcentuales 
globalmente, pero el beneficio fue heterogéneo: mayor para el perfil 
'en riesgo' (+25pp) que para 'alto rendimiento' (+8pp). 

Limitaciones: el modelo asume que las variables de desempeño están 
disponibles tempranamente, lo cual puede no ser realista para intervención 
temprana. Además, factores sociales y contextuales no capturados podrían 
mejorar la predicción."
```

---

## Final Exam Tips

### Time Management
- **Read all missions first** (5 min)
- **Start with data preparation** (Mission 1) — get this right and the rest flows
- **Don't over-optimize** — 2 models compared is enough, don't try 10
- **Leave time for interpretation** — numbers without explanation get fewer points

### What Gets Points
✅ **Correct workflow** (no data leakage, proper split)
✅ **Justification** (why you chose that K, that model, that feature)
✅ **Interpretation** (what do the clusters mean? what do the results tell us?)
✅ **Comparison** (model A vs B, original vs reduced features)
✅ **Visualizations** (elbow plot, cluster scatter, performance comparison)

### What Loses Points
❌ Data leakage (fitting scaler on all data)
❌ Using test set for decisions
❌ No justification (just showing numbers)
❌ Code without output (cells not executed)
❌ Ignoring the questions (exercise asks for 3 derived variables, you create 1)

### Common Traps
- **"Compare representations"** → Must cluster on original AND reduced, then compare
- **"At least two models"** → Minimum 2, but 3 is safer
- **"Justify your decision"** → Always explain WHY you chose X over Y
- **"Use appropriate metrics"** → Imbalanced classes? Use balanced accuracy

### Quick Reference: What to Import
```python
# Always needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Supervised models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC

# Unsupervised
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering

# Metrics
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    silhouette_score
)
```

---

## Good luck! 🎯

**Remember:**
- Supervised = predict a target (regression or classification)
- Unsupervised = find structure (clustering or dimensionality reduction)
- Train → Validation → Test (use validation to decide, test to evaluate once)
- No data leakage (fit on train only)
- Justify your choices (why this K? why this model?)
- Interpret your results (what do the numbers mean?)
