/*+ HashJoin(cn t mc k mk)
 HashJoin(t mc k mk)
 HashJoin(mc k mk)
 NestLoop(k mk)
 SeqScan(cn)
 SeqScan(t)
 SeqScan(mc)
 IndexScan(k)
 IndexScan(mk)
 Leading((cn (t (mc (k mk))))) */
SELECT MIN(t.title) AS movie_title
FROM company_name AS cn,
     keyword AS k,
     movie_companies AS mc,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code ='[us]'
  AND k.keyword ='character-name-in-title'
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND mc.movie_id = mk.movie_id;

