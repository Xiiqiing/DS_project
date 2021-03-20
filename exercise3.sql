--1.List the authors who wrote articles classified as showing extreme bias scrapped from the domain charismanews.com (Links to an external site.) on January 25, 2018.  

--select AU.authors
--from author AU
--where AU.aut_id in (select W.aut_id
--					from wrote W
--					where W.art_id in (select AR.art_id
--									   from article AR, source S
--									   where AR.type = 'bias' and AR.scraped_at = '2018-01-25' and S.domain = 'charismanews.com'))


-- 2.For each type in the dataset (fake, satire, bias, etc), list the type and the number of articles with that type.

--select AR.type, count(*) as count
--from article AR
--group by AR.type


-- 3.List the titles of the articles authored by Hamilton Strategies except for the ones that have the meta keyword ’president trump’ or have been tagged as ’Donald Trump’.

--select AR.title
--from article AR, author AU
--where AU.authors = 'Hamilton Strategies' and AU.aut_id in (select W.aut_id
--														   from wrote W
--													   where W.art_id in (select AR.art_id
--																			  from article AR
--																			  except
--																			  select AR.art_id
--																			  from article AR
--																			  where AR.meta_keywords = 'president trump' and AR.tags = 'Donald Trump'))

--4. Let P be the set of tags containing the string ’energies’. List the IDs of articles whose tags include all tags in P.

--select AR.art_id
--from article AR
--where AR.tags in (select AR.tags
--				  from article AR
--				  where AR.tags like '_%energies_%')