/*+ HashJoin(k t cn mc mk)
 NestLoop(t cn mc mk)
 HashJoin(t cn mc)
 NestLoop(cn mc)
 IndexScan(k)
 SeqScan(t)
 SeqScan(cn)
 IndexScan(mc)
 IndexScan(mk)
 Leading((k ((t (cn mc)) mk))) */
SELECT MIN(t.title) AS movie_title
FROM company_name AS cn,
     keyword AS k,
     movie_companies AS mc,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code ='[sm]'
  AND k.keyword ='character-name-in-title'
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND mc.movie_id = mk.movie_id;

