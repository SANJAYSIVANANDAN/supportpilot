CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    email VARCHAR(320) UNIQUE NOT NULL,
    phone VARCHAR(30),
    company VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tickets (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    subject VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
    assigned_to VARCHAR(200),
    resolution TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);

INSERT INTO customers (name, email, phone, company) VALUES
    ('Alice Johnson', 'alice@example.com', '+1-555-0101', 'Acme Corp'),
    ('Bob Smith', 'bob@example.com', '+1-555-0102', 'Globex Inc'),
    ('Carol White', 'carol@example.com', '+1-555-0103', NULL)
ON CONFLICT (email) DO NOTHING;

INSERT INTO tickets (customer_id, subject, description, status, priority) VALUES
    (1, 'Cannot login to dashboard', 'I keep getting a 403 error when trying to access the analytics dashboard.', 'open', 'high'),
    (1, 'Billing discrepancy', 'My last invoice shows a charge I do not recognise.', 'open', 'medium'),
    (2, 'Feature request: dark mode', 'Would love a dark-mode toggle in settings.', 'open', 'low'),
    (3, 'App crashes on upload', 'Uploading a CSV larger than 10 MB causes the app to crash.', 'in_progress', 'urgent')
ON CONFLICT DO NOTHING;
