### **Instructions for Participants: AIAC2.0 Code Submission**

Welcome to the **AI Alphas Competition 2.0 (AIAC2.0)**! Below are detailed instructions to help you use the provided template to interact with the WorldQuant Brain API and a Large Language Model (LLM) to generate alpha expressions directly from research ideas and hypotheses.

---

#### **Code Template**
The provided Python template includes the following main functions:

1. **`call_llm(messages, output_structure)`**
   - Interface with your preferred LLM to process prompts.
   - Replace the example OpenAI implementation with your own LLM logic.

2. **`get_operators_reference(brain_session)`**
   - Retrieves available operators from BRAIN API for LLM reference.

3. **`get_dataset_reference(brain_session, dataset_id, region, universe)`**
   - Retrieves dataset and datafield information from BRAIN API.

4. **`generate_alpha_expressions(hypothesis, operators_reference, dataset_reference, num_alphas)`**
   - Generates new Alpha expressions based on research hypothesis using operators and datafields.

---

#### **Steps to Follow**

##### **Step 1: Setup**
1. Install the required Python libraries:
   ```bash
   pip install pandas tabulate openai pydantic
   ```
2. Replace code of the call_llm function that is compliant with the usage of LLM of your choice.
---

##### **Step 2: Prepare References**
1. Get operators and dataset references:
   ```python
   operators_reference = get_operators_reference(s)
   dataset_reference = get_dataset_reference(s, dataset_id="model110")
   ```
2. These references will be passed to your LLM to generate valid alpha expressions. You should work on systematic selection of appropriate datasets and operators as per your hypothesis. 
You can either build your own solution using other helper functions or simply prompt LLM for this task, use your innovation :)

---

##### **Step 3: Generate Alpha Expressions**
1. Define your research hypothesis or investment idea (can be from papers, blogs, reports, or your own ideas):
   ```python
   hypothesis = "Your research idea here"
   ```
2. Generate alpha expressions using the hypothesis:
   ```python
   result = generate_alpha_expressions(hypothesis, operators_reference, dataset_reference, num_alphas=10)
   ```
3. The function will send your hypothesis to the LLM and return structured alpha expressions with economic rationale.
4. Edit the `generate_alpha_expressions` function to refine prompts and customize output.

---

##### **Step 4: Simulate, Tag and Add Description to Alphas**
1. Use the ACE Library to simulate the generated Alphas.
2. Add LLM tag to each Alpha to indicate it was LLM-generated. This LLM tag should be among the list of valid LLM tags that we have.
3. Add description to each Alpha from the generated economic rationale.

---

##### **Step 5: Validate output format**
Make sure that your notebook generates output in the form of Alphas.json generated in the aiac2.0_sample.ipynb. We have also attached the json file as part of this template folder.
Feel free to change or add any other functions as per your own research process.

---

#### **Important Notes**
- **Research Input:** Use any text as input (papers, blogs, reports, notes or your own hypothesis)
- **Scoring:** AIAC2.0 scoring is based directly on Alpha submissions.
- **Tagging:** Ensure all Alphas have an LLM tag and description.
- **LLM Usage:** You may use any LLM of your choice, but ensure tagging from the list of LLM tags available
- **Alpha Validity:** All expressions must compile, simulate successfully, and comply with BRAIN and competition rules.

---

#### **Key Dates**
- **January 19, 2026:** Submissions start
- **February 15, 2026:** Submissions end

---

#### **Support**
For any questions or technical issues, refer to the **ACE Library Documentation**, **FAQs**, **Competition Guidelines** or contact the **WorldQuant BRAIN Team**.

Good luck, and we look forward to seeing your innovative submissions!