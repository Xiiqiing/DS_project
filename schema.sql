CREATE TABLE article
(
  art_id INT NOT NULL,
  title text NOT NULL,
  type text NOT NULL,
  content text NOT NULL,
  scraped_at DATE NOT NULL,
  meta_keywords text NOT NULL,
  tags text NOT NULL,
  PRIMARY KEY (art_id)
);

CREATE TABLE author
(
  aut_id double precision NOT NULL,
  authors text NOT NULL,
  PRIMARY KEY (aut_id)
);

CREATE TABLE source
(
  source_id double precision NOT NULL,
  URL text NOT NULL,
  domain text NOT NULL,
  art_id INT NOT NULL,
  PRIMARY KEY (art_id),
  FOREIGN KEY (art_id) REFERENCES article(art_id)
);

CREATE TABLE wrote
(
  aut_id double precision NOT NULL,
  art_id INT NOT NULL,
  PRIMARY KEY (art_id),
  FOREIGN KEY (aut_id) REFERENCES author(aut_id),
  FOREIGN KEY (art_id) REFERENCES article(art_id)
);