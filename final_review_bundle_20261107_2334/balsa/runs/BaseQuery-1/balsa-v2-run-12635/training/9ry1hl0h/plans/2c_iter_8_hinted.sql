/*+ HashJoin(cn mc t k mk)
 HashJoin(mc t k mk)
 HashJoin(t k mk)
 NestLoop(k mk)
 SeqScan(cn)
 SeqScan(mc)
 SeqScan(t)
 IndexScan(k)
 IndexScan(mk)
 Leading((cn (mc (t (k mk))))) */
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

