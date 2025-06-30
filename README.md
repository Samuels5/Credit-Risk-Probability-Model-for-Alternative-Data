# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to have a robust understanding of their risk profiles and to quantify them accurately. This necessitates models that are not only predictive but also interpretable and well-documented. An interpretable model allows regulators and stakeholders to understand how it arrives at its conclusions, ensuring transparency and justifiability in risk assessment. Comprehensive documentation provides a clear audit trail of the model's development, validation, and implementation, which is crucial for regulatory compliance.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

The dataset is from an eCommerce platform and tracks customer transactions, not loan performance, so it lacks a direct "default" label. Therefore, a proxy variable is necessary to represent credit risk. We can infer risk by analyzing customer behavior, such as Recency, Frequency, and Monetary (RFM) value.

The primary business risk of using a proxy is that it is an indirect measure. A customer identified as "high-risk" due to low engagement might not be a genuine credit risk, leading to the denial of credit to potentially good customers (false positives) and resulting in lost business opportunities. Conversely, an engaged user might still pose a credit risk (false negative), which could lead to financial losses.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

**Simple, Interpretable Model (e.g., Logistic Regression with Weight of Evidence):**

- **Pros:** These models are highly interpretable and easy to explain to stakeholders and regulators. The use of WoE transforms features into a linear scale, making the impact of each variable on the outcome clear. They are also computationally less expensive and faster to train.
- **Cons:** They may not capture complex, non-linear relationships in the data, potentially leading to lower predictive accuracy compared to more complex models.

**Complex, High-Performance Model (e.g., Gradient Boosting):**

- **Pros:** These models can capture intricate patterns and interactions in the data, often resulting in higher predictive accuracy.
- **Cons:** Their "black-box" nature makes them difficult to interpret. It is challenging to explain why the model made a specific prediction, which is a significant drawback in a regulated financial context where model transparency is essential. They are also more computationally expensive and can be prone to overfitting if not carefully tuned.
