/*+ HashJoin(cn mc k t mk)
 HashJoin(mc k t mk)
 HashJoin(k t mk)
 NestLoop(t mk)
 SeqScan(cn)
 SeqScan(mc)
 IndexScan(k)
 SeqScan(t)
 IndexScan(mk)
 Leading((cn (mc (k (t mk))))) */
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

