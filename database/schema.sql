-- ============================================================
-- Bengaluru Two-Wheeler Rider Skin Health & Sun Exposure Advisor
-- MySQL Database Schema
-- Compatible with IBM Cloud Databases for MySQL
-- ============================================================

CREATE DATABASE IF NOT EXISTS bengaluru_rider_advisor
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE bengaluru_rider_advisor;

-- ============================================================
-- riders: registered commuter profiles
-- ============================================================
CREATE TABLE IF NOT EXISTS riders (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    name             VARCHAR(100)  NOT NULL,
    email            VARCHAR(150)  UNIQUE,
    phone            VARCHAR(20),
    age              TINYINT UNSIGNED,
    -- skin profile
    skin_type        ENUM('oily','dry','combination','sensitive','normal') NOT NULL DEFAULT 'normal',
    skin_sensitivity ENUM('low','moderate','high','hyper_reactive')        NOT NULL DEFAULT 'moderate',
    -- existing protection habits stored as a JSON blob
    existing_habits  JSON COMMENT '{"uses_sunscreen":true,"spf_rating":"SPF50+","reapplies_sunscreen":false,"wears_arm_sleeves":true,"wears_full_jacket":false,"visor_type":"clear","wears_gloves":false,"uses_face_mask_gaiter":false}',
    -- preferred commute defaults
    default_route    VARCHAR(200),
    default_duration DECIMAL(4,2) COMMENT 'hours',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_skin_sensitivity (skin_sensitivity)
) ENGINE=InnoDB;

-- ============================================================
-- exposure_logs: per-trip advisory records
-- ============================================================
CREATE TABLE IF NOT EXISTS exposure_logs (
    id                    BIGINT AUTO_INCREMENT PRIMARY KEY,
    rider_id              INT,
    -- trip context
    route                 VARCHAR(200) NOT NULL,
    departure_time        DATETIME     NOT NULL,
    duration_hours        DECIMAL(4,2) NOT NULL,
    -- real-time telemetry snapshot
    uv_index              DECIMAL(5,2),
    temperature_c         DECIMAL(5,2),
    humidity_pct          DECIMAL(5,2),
    precipitation_prob    TINYINT UNSIGNED,
    pm25                  DECIMAL(7,2),
    dust                  DECIMAL(7,2),
    -- calculated scores
    net_exposure_score    DECIMAL(8,4),
    sensitivity_multiplier DECIMAL(4,2),
    protection_discount   DECIMAL(4,2),
    -- AI outputs
    travel_verdict        ENUM('SAFE_TO_TRAVEL','TRAVEL_WITH_GEAR','DELAY_RECOMMENDED','AVOID_NON_ESSENTIAL') NOT NULL,
    best_travel_window    VARCHAR(500) COMMENT 'JSON array of top-3 windows',
    protection_gap_advice TEXT,
    skincare_advice       TEXT,
    ai_recommendations    JSON COMMENT 'Full structured LLM output per agent',
    -- LLM governance
    llm_provider_used     VARCHAR(50)  COMMENT 'granite|groq|gemini',
    llm_model_used        VARCHAR(100),
    prompt_tokens         INT UNSIGNED DEFAULT 0,
    completion_tokens     INT UNSIGNED DEFAULT 0,
    total_cost_usd        DECIMAL(10,6) DEFAULT 0.000000,
    -- metadata
    logged_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rider_id) REFERENCES riders(id) ON DELETE SET NULL,
    INDEX idx_rider_id    (rider_id),
    INDEX idx_verdict     (travel_verdict),
    INDEX idx_logged_at   (logged_at)
) ENGINE=InnoDB;

-- ============================================================
-- agent_chat_history: Interactive AI Agent Console logs
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_chat_history (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    rider_id     INT,
    session_id   VARCHAR(64)  NOT NULL,
    agent_name   VARCHAR(100) NOT NULL,
    role         ENUM('user','assistant') NOT NULL,
    message      TEXT NOT NULL,
    tokens_used  INT UNSIGNED DEFAULT 0,
    cost_usd     DECIMAL(10,6) DEFAULT 0.000000,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY  (rider_id) REFERENCES riders(id) ON DELETE SET NULL,
    INDEX idx_session (session_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- ============================================================
-- Seed Data — Bengaluru commuter profiles
-- ============================================================
INSERT INTO riders (name, email, skin_type, skin_sensitivity, existing_habits, default_route, default_duration) VALUES
('Arjun Sharma',   'arjun@example.com',   'oily',        'moderate',      '{"uses_sunscreen":true,"spf_rating":"SPF50+","reapplies_sunscreen":false,"wears_arm_sleeves":false,"wears_full_jacket":false,"visor_type":"clear","wears_gloves":false,"uses_face_mask_gaiter":true}',  'Electronic City – Indiranagar', 1.25),
('Priya Nair',     'priya@example.com',   'sensitive',   'hyper_reactive','{"uses_sunscreen":true,"spf_rating":"SPF50+","reapplies_sunscreen":true,"wears_arm_sleeves":true,"wears_full_jacket":false,"visor_type":"uv_tinted","wears_gloves":true,"uses_face_mask_gaiter":true}',   'Whitefield – Koramangala ORR',  1.50),
('Kiran Reddy',    'kiran@example.com',   'combination', 'high',          '{"uses_sunscreen":false,"spf_rating":"none","reapplies_sunscreen":false,"wears_arm_sleeves":false,"wears_full_jacket":true,"visor_type":"clear","wears_gloves":true,"uses_face_mask_gaiter":false}',       'Hebbal – MG Road',              1.00),
('Deepa Menon',    'deepa@example.com',   'dry',         'moderate',      '{"uses_sunscreen":true,"spf_rating":"SPF30","reapplies_sunscreen":false,"wears_arm_sleeves":true,"wears_full_jacket":false,"visor_type":"clear","wears_gloves":false,"uses_face_mask_gaiter":false}',      'BTM Layout – Marathahalli',     1.75),
('Suresh Kumar',   'suresh@example.com',  'normal',      'low',           '{"uses_sunscreen":false,"spf_rating":"none","reapplies_sunscreen":false,"wears_arm_sleeves":false,"wears_full_jacket":false,"visor_type":"clear","wears_gloves":false,"uses_face_mask_gaiter":false}',      'JP Nagar – Bellandur',          0.75);

-- Sample exposure log
INSERT INTO exposure_logs (rider_id, route, departure_time, duration_hours, uv_index, temperature_c, humidity_pct, precipitation_prob, pm25, dust, net_exposure_score, sensitivity_multiplier, protection_discount, travel_verdict, best_travel_window, protection_gap_advice, skincare_advice, llm_provider_used, llm_model_used, prompt_tokens, completion_tokens, total_cost_usd) VALUES
(1, 'Electronic City – Indiranagar', '2026-11-19 09:00:00', 1.25, 8.5, 29.0, 68.0, 20, 65.0, 12.0, 11.3400, 1.00, 0.30, 'TRAVEL_WITH_GEAR', '["07:00-08:00 (UV 3.2, PM2.5 38)","18:00-19:00 (UV 1.1, PM2.5 42)","06:30-07:30 (UV 2.1, PM2.5 35)"]', 'Missing: UPF arm sleeves (-15%), UV-tinted visor (-10%). Add SPF50+ reapplication every 2h.', 'Apply niacinamide serum before ride. Post-ride: micellar cleanse, hyaluronic moisturizer.', 'granite', 'ibm/granite-3-3-8b-instruct', 512, 380, 0.000041);
