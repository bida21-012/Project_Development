from flask import Flask, jsonify, request
import random
import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def generate_data():
    print("Generating data...")  # Debug log
    countries_regions = {
        'USA': ['North America', 'Central USA'],
        'Canada': ['North America', 'Western Canada'],
        'UK': ['Northern Europe', 'Western Europe'],
        'Germany': ['Central Europe', 'Western Europe'],
        'India': ['South Asia', 'Central India'],
        'Japan': ['East Asia', 'Pacific Rim'],
        'Brazil': ['South America', 'Latin America'],
        'Mexico': ['Central America', 'North America'],
        'Australia': ['Oceania', 'Australasia'],
        'South Africa': ['Southern Africa', 'Sub-Saharan Africa']
    }

    job_types = [
        'Software Developer', 'Data Scientist', 'Sales Engineer', 'Product Manager', 'UX Designer',
        'Systems Analyst', 'QA Tester', 'DevOps Engineer', 'Business Analyst', 'Cloud Architect',
        'Security Specialist', 'AI Engineer', 'Database Admin', 'IT Support', 'Network Engineer',
        'Technical Writer', 'Mobile Developer', 'Game Developer', 'Data Engineer', 'Machine Learning Scientist'
    ]

    service_types = ['SaaS', 'Consulting', 'Support', 'Cloud Storage', 'Analytics', 'Monitoring', 'Training', 'Security', 'AI Platform', 'IoT']
    product_names = [
        'CloudSync Pro', 'DataVault Analytics', 'AI-Predictor', 'SecureNet Firewall', 'StreamFlow CRM',
        'IntelliSuite ERP', 'Visionary AI', 'NexGen Database', 'SmartTrack IoT', 'Quantum Compute',
        'FlowCore SaaS', 'Insight360 Dashboard', 'CyberGuard VPN', 'OptiChain SCM', 'TrueView BI',
        'SkyLink Cloud', 'PulseMonitor', 'DataForge ETL', 'AI-Connect Chat', 'BrightPath LMS'
    ]
    request_categories = ['Technical', 'Billing', 'General', 'Sales', 'Feedback', 'Bug Report', 'Feature Request']
    action_types = ['Viewed', 'Clicked', 'Downloaded', 'Subscribed', 'Shared']
    browsers = ['Chrome', 'Firefox', 'Edge', 'Safari', 'Opera', 'Brave']
    responses = ['Success', 'Error', 'Timeout', 'Redirect']
    referral_sources = ['Google', 'LinkedIn', 'Email Campaign', 'Facebook', 'Twitter', 'Direct', 'Partner']
    assistants = ['ChatGPT', 'Alexa', 'Google Assistant', 'Cortana', 'Siri']
    customer_behaviours = ['Loyal', 'Churned', 'New', 'Frequent Buyer', 'Occasional', 'Inactive']
    company_names = ['TechNova', 'DataSync', 'CloudCore', 'InnoWare', 'NetGenius', 'ByteWorks', 'SoftEdge', 'Nexon', 'CodeCrafters', 'SysNova']

    data = []

    now = datetime.datetime.now()

    for i in range(10000):  
        country = random.choice(list(countries_regions.keys()))
        region = random.choice(countries_regions[country])

        job = {
            "Job_ID": i + 1,
            "Company_Name": random.choice(company_names),
            "Country": country,
            "Region": region,
            "Job_Type": random.choice(job_types),
            "Service_Type": random.choice(service_types),
            "Revenue": round(random.uniform(500, 10000), 2),
            "Action_Type": random.choice(action_types),
            "Product_Name": random.choice(product_names),
            "Request_Category": random.choice(request_categories),
            "User_ID": f"user_{random.randint(1000, 9999)}",
            "IP_Address": f"{random.randint(10, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "Pages_Per_Session": random.randint(1, 10),
            "Session_Duration": round(random.uniform(1, 30), 2),
            "Transaction": random.randint(1, 5),
            "Profit": round(random.uniform(100, 9000), 2),
            "Referral": random.choice(referral_sources),
            "AI_Assistant": random.choice(assistants),
            "Engaged_User": random.choice([True, False]),
            "Browser": random.choice(browsers),
            "Response": random.choice(responses),
            "Request_URL": f"/api/v1/{random.choice(['get', 'post', 'update', 'delete'])}/resource_{random.randint(1, 20)}",
            "Conversion_Status": random.choice([True, False]),
            "Demo_Request": random.choice([True, False]),
            "Customer_Behavior": random.choice(customer_behaviours),
            "Number_of_Sales": random.randint(1, 100),
            "Date": (now - datetime.timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
            "Timestamp": now.isoformat(),
            "Promotional_Event": random.choices([True, False], weights=[0.2, 0.8])[0]  # 20% chance of True
        }
        data.append(job)

    print(f"Generated {len(data)} entries")  # Debug log
    return data

@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        country = request.args.get('country')
        print(f"Received request with country filter: {country}")  # Debug log
        data = generate_data()

        if country:
            data = [entry for entry in data if entry['Country'].lower() == country.lower()]
            print(f"Filtered data for {country}: {len(data)} entries")  # Debug log

        print("Returning response...")  # Debug log
        return jsonify(data)
    except Exception as e:
        print(f"Error in /get_data: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)