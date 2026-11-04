/*+ NestLoop(mc t k mk cn)
 HashJoin(mc t k mk)
 HashJoin(t k mk)
 NestLoop(k mk)
 SeqScan(mc)
 SeqScan(t)
 IndexScan(k)
 IndexScan(mk)
 IndexScan(cn)
 Leading(((mc (t (k mk))) cn)) */
SELECT MIN(t.title) AS movie_title
FROM company_name AS cn,
     keyword AS k,
     movie_companies AS mc,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code ='[nl]'
  AND k.keyword ='character-name-in-title'
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND mc.movie_id = mk.movie_id;

