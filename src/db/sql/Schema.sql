CREATE TABLE Topic
(
    ID   INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL UNIQUE CHECK (length(Name) <= 500)
);

INSERT INTO Topic (Name)
VALUES ('Unknown');

CREATE TABLE File
(
    ID               INTEGER PRIMARY KEY AUTOINCREMENT,
    Name             TEXT    NOT NULL,
    TopicID          INTEGER,
    Extension        TEXT,
    Path             TEXT,
    Rate             INTEGER,
    MinFreq          REAL,
    MaxFreq          REAL,
    BitDepth         INTEGER,
    Channels         INTEGER,
    Duration         REAL,
    RMSLoudness      REAL,
    ZeroCrossingRate REAL,
    SpectralCentroid REAL,
    EQ_20_250_Hz     REAL,
    EQ_250_2000_Hz   REAL,
    EQ_2000_6000_Hz  REAL,
    EQ_6000_20000_Hz REAL,
    MFCC_1           REAL,
    MFCC_2           REAL,
    MFCC_3           REAL,
    MFCC_4           REAL,
    MFCC_5           REAL,
    MFCC_6           REAL,
    MFCC_7           REAL,
    MFCC_8           REAL,
    MFCC_9           REAL,
    MFCC_10          REAL,
    MFCC_11          REAL,
    MFCC_12          REAL,
    MFCC_13          REAL,
    Summary          TEXT    NOT NULL,
    Conflict         INTEGER NOT NULL CHECK (Conflict IN (0, 1)),
    Silence          REAL    NOT NULL,

    FOREIGN KEY (TopicID) REFERENCES Topic (ID)
);

CREATE TABLE Utterance
(
    ID        INTEGER PRIMARY KEY AUTOINCREMENT,
    FileID    INTEGER                                                       NOT NULL,
    Speaker   TEXT CHECK (Speaker IN ('Customer', 'CSR'))                   NOT NULL,
    Sequence  INTEGER                                                       NOT NULL,
    StartTime REAL                                                          NOT NULL,
    EndTime   REAL                                                          NOT NULL,
    Content   TEXT                                                          NOT NULL,
    Sentiment TEXT CHECK (Sentiment IN ('Neutral', 'Positive', 'Negative')) NOT NULL,
    Profane   INTEGER                                                       NOT NULL CHECK (Profane IN (0, 1)),
    FOREIGN KEY (FileID) REFERENCES File (ID)
);
