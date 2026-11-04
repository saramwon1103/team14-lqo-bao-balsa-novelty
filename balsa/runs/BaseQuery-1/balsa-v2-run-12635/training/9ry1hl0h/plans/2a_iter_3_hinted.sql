/*+ HashJoin(cn k mk t mc)
 NestLoop(k mk t mc)
 MergeJoin(k mk t)
 HashJoin(mk t)
 SeqScan(cn)
 IndexScan(k)
 SeqScan(mk)
 SeqScan(t)
 IndexScan(mc)
 Leading((cn ((k (mk t)) mc))) */
SELECT MIN(t.title) AS movie_title
FROM company_name AS cn,
     keyword AS k,
     movie_companies AS mc,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code ='[de]'
  AND k.keyword ='character-name-in-title'
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND mc.movie_id = mk.movie_id;

