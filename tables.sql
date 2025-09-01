-- create_tables.sql 파일 내용

-- Cosmetics 테이블 생성
CREATE TABLE Cosmetics (
    prduct_nm VARCHAR(255) NOT NULL,
    cmpnm VARCHAR(255),
    registration_date DATE,
    PRIMARY KEY (prduct_nm)
);

-- Ingredients 테이블 생성
CREATE TABLE Ingredients (
    ingd_nm VARCHAR(255) NOT NULL,
    ingd_eng_nm VARCHAR(255),
    ingd_description TEXT,
    PRIMARY KEY (ingd_nm)
);

-- Cosmetics_Ingredients 테이블 생성 (조인 테이블)
CREATE TABLE Cosmetics_Ingredients (
    prduct_nm VARCHAR(255) NOT NULL,
    ingd_nm VARCHAR(255) NOT NULL,
    ingd_cntnt VARCHAR(255),
    PRIMARY KEY (prduct_nm, ingd_nm),
    FOREIGN KEY (prduct_nm) REFERENCES Cosmetics (prduct_nm),
    FOREIGN KEY (ingd_nm) REFERENCES Ingredients (ingd_nm)
);