CREATE TABLE IF NOT EXISTS clsf_type (
    class varchar(64) PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS detection (
    detection_id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    class varchar(64) NOT NULL REFERENCES clsf_type(class),
    image_path varchar(256),
    xmin DOUBLE PRECISION,
    ymin DOUBLE PRECISION,
    xmax DOUBLE PRECISION,
    ymax DOUBLE PRECISION,
    confidence DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    detection_dtm TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
