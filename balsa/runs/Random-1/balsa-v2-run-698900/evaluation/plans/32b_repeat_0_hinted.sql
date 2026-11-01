/*+ HashJoin(k t1 ml t2 lt mk)
 NestLoop(t1 ml t2 lt mk)
 NestLoop(t1 ml t2 lt)
 HashJoin(t1 ml t2)
 MergeJoin(ml t2)
 IndexScan(k)
 IndexScan(t1)
 SeqScan(ml)
 SeqScan(t2)
 IndexScan(lt)
 IndexScan(mk)
 Leading((k (((t1 (ml t2)) lt) mk))) */
SELECT MIN(lt.link) AS link_type,
       MIN(t1.title) AS first_movie,
       MIN(t2.title) AS second_movie
FROM keyword AS k,
     link_type AS lt,
     movie_keyword AS mk,
     movie_link AS ml,
     title AS t1,
     title AS t2
WHERE k.keyword ='character-name-in-title'
  AND mk.keyword_id = k.id
  AND t1.id = mk.movie_id
  AND ml.movie_id = t1.id
  AND ml.linked_movie_id = t2.id
  AND lt.id = ml.link_type_id
  AND mk.movie_id = t1.id;

